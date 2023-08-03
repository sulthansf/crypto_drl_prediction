import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from finta import TA
from binance.client import Client


class PredictionGameDRLPlayer:
    def __init__(self, api_key, api_secret, symbol, interval, features, ta_period, window_size, prediction_period, action_space, q_network_path, scaler_path, verbose=1, logging=False, log_path=None):
        """
        Initialize the PredictionGameDRLPlayer.

        Args:
            api_key (str): The API key for the Binance account.
            api_secret (str): The API secret for the Binance account.
            symbol (str): The symbol to trade.
            interval (str): The interval for the price data.
            features (list): The list of features to use for training.
            ta_period (int): The period used for calculating technical indicators.
            window_size (int): The size of the window for the state.
            prediction_period (int): The period used for predicting the price.
            action_space (list): The list of actions.
            q_network_path (str): The path to the pre-trained Q-network.
            scaler_path (str): The path to the scaler.
            verbose (int): The verbosity level (0, 1 or 2).
            logging (bool): Indicates whether to log the training process.
            log_path (str): The path to the log file.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.interval = interval
        self.features = features
        self.ta_period = ta_period
        self.window_size = window_size
        self.prediction_period = prediction_period
        self.state_shape = (window_size, len(features))
        self.action_space = action_space
        self.verbose = verbose
        self.logging = logging
        if log_path:
            self.log_path = log_path
        else:
            self.log_path = '../../log/env_log_' + \
                time.strftime("%Y%m%d_%H%M%S") + '.txt'

        # Load the pre-trained Q-network
        self.q_network = tf.keras.models.load_model(q_network_path)

        # Load the scaler
        self.scaler = joblib.load(scaler_path)

        # Create the Binance client
        self.client = Client(self.api_key, self.api_secret)

        # Set the api interval
        if self.interval == 1:
            self.api_interval = Client.KLINE_INTERVAL_1MINUTE
        elif self.interval == 3:
            self.api_interval = Client.KLINE_INTERVAL_3MINUTE
        elif self.interval == 5:
            self.api_interval = Client.KLINE_INTERVAL_5MINUTE
        elif self.interval == 15:
            self.api_interval = Client.KLINE_INTERVAL_15MINUTE
        elif self.interval == 30:
            self.api_interval = Client.KLINE_INTERVAL_30MINUTE
        elif self.interval == 60:
            self.api_interval = Client.KLINE_INTERVAL_1HOUR
        else:
            raise ValueError(
                'The interval is in minutes and must be 1, 3, 5, 15, 30 or 60.')

    def get_current_state(self):
        """
        Get the current state.

        Returns:
            state (np.ndarray): The current state.
        """
        # Get the latest price data
        data_period = 2 * (self.window_size + self.ta_period)
        klines = self.client.get_historical_klines(
            self.symbol, self.interval, str(data_period) + " min ago UTC")
        klines_df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        dataset_df = klines_df[['open', 'high',
                                'low', 'close', 'volume']].astype(float)
        dataset = self.process_data(dataset_df, self.ta_period)
        state = dataset[-self.window_size:]
        if state.isnan().any():
            raise ValueError('The state contains NaN values.')
        else:
            return state

    def process_data(self, dataset_df, ta_period):
        """
        Process the dataset by calculating technical indicators and scaling the data.

        Args:
            dataset_df (pd.DataFrame): The input dataset as a pandas DataFrame.
            ta_period (int): The period used for calculating technical indicators.

        Returns:
            dataset (np.ndarray): The processed dataset.
        """
        dataset_df['sma'] = TA.SMA(dataset_df, ta_period)
        dataset_df['ema'] = TA.EMA(dataset_df, ta_period)
        dataset_df['dema'] = TA.DEMA(dataset_df, ta_period)
        dataset_df['mom'] = TA.MOM(dataset_df, ta_period)
        dataset_df['vzo'] = TA.VZO(dataset_df, ta_period)
        dataset_df[['macd', 'signal']] = TA.MACD(dataset_df)
        dataset_df['rsi'] = TA.RSI(dataset_df, ta_period)
        dataset_df[['bb_upper', 'bb_middle', 'bb_lower']
                   ] = TA.BBANDS(dataset_df, ta_period)
        dataset_df['stoch_k'] = TA.STOCH(dataset_df, ta_period)
        dataset_df['stoch_d'] = TA.STOCHD(dataset_df, ta_period)
        dataset_df['er'] = TA.ER(dataset_df, ta_period)
        dataset_df[['ppo', 'ppo_sig', 'ppo_hist']
                   ] = TA.PPO(dataset_df, ta_period)
        dataset_df['roc'] = TA.ROC(dataset_df, ta_period)
        dataset_df['atr'] = TA.ATR(dataset_df, ta_period)
        dataset_df['sar'] = TA.SAR(dataset_df, ta_period)
        dataset_df[['mobo_upper', 'mobo_middle', 'mobo_lower']
                   ] = TA.MOBO(dataset_df, ta_period)
        dataset_df[['dip', 'din']] = TA.DMI(dataset_df, ta_period)
        dataset_df[['tsi', 'tsi_signal']] = TA.TSI(dataset_df, ta_period)
        dataset_df['tp'] = TA.TP(dataset_df)
        dataset_df['adl'] = TA.ADL(dataset_df)
        dataset_df[['basp_buy', 'basp_sell']] = TA.BASP(dataset_df, ta_period)
        num_drop = ta_period
        dataset_df = dataset_df.iloc[num_drop:]
        dataset = dataset_df[self.features].to_numpy().astype(np.float32)
        dataset_scaled = self.scaler.transform(dataset).astype(np.float32)
        return dataset_scaled

    def choose_action(self, state):
        """
        Choose an action based on the current state.

        Args:
            state (np.ndarray): The current state.

        Returns:
            action (float): The chosen action.
        """
        if state.shape == self.state_shape:
            state = self.scaler.transform(state)
            return self.action_space[np.argmax(self.q_network.predict(np.array([state]), verbose=0))]
        else:
            raise ValueError(
                'The state shape must be ' + str(self.state_shape) + '.')

    def next_action(self):
        """
        Get the next action.

        Returns:
            action (float): The next action.
        """
        state = self.get_current_state()
        return self.choose_action(state)
