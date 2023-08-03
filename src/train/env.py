import random
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from collections import deque
from finta import TA


class PredictionGameEnvironment:
    """
    A prediction game environment for the RL agent.
    """

    def __init__(self, dataset_df, features, sampling_period, ta_period, window_size, episode_length, eval_episode_length, prediction_period, scaler_path=None, verbose=1, logging=False, log_path=None):
        """
        Initialize the environment.

        Args:
            dataset_df (pd.DataFrame): The input dataset as a pandas DataFrame.
            features (list): The list of features to use for training.
            sampling_period (int): The sampling period of the dataset.
            ta_period (int): The period used for calculating technical indicators.
            window_size (int): The size of the window for the state.
            episode_length (int): The length of an episode.
            eval_episode_length (int): The length of the evaluation period.
            prediction_period (int): The period used for predicting the price.
            scaler_path (str): The path to the scaler file.
            verbose (int): The verbosity level (0, 1 or 2).
            logging (bool): Indicates whether to log the training process.
            log_path (str): The path to the log file.
        """
        self.features = features
        self.sampling_period = sampling_period
        price_idx = self.features.index('close')
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.dataset, self.price_data = self.process_data(
            dataset_df, ta_period, price_idx)
        self.num_data = len(self.dataset)
        self.window_size = window_size
        self.episode_length = episode_length
        self.eval_episode_length = eval_episode_length
        self.prediction_period = prediction_period
        self.action_space = [-1.0, 0.0, 1.0]
        self.reward_space = [-1.0, 0.0, 0.7]
        self.prev_episode_lengths = deque(maxlen=25)
        self.verbose = verbose
        self.logging = logging
        if log_path:
            self.log_path = log_path
        else:
            self.log_path = '../../log/env_log_' + \
                time.strftime("%Y%m%d_%H%M%S") + '.txt'
        self.reset()

    def reset(self, eval=False):
        """
        Reset the environment for a new episode and return the initial state.

        Args:
            eval (bool): Indicates whether the current episode is used for evaluation or not.

        Returns:
            state (np.ndarray): The initial state of the environment.
        """
        self.eval_episode = eval
        self.done = False
        self.step_count = 0
        self.balance = 10.0 if not self.eval_episode else 25.0
        self.reward = None
        self.action = None
        self.winning_action = None
        self.current_price = None
        self.prediction_price = None
        self.action_results_count = [0.0, 0.0, 0.0]
        self.state_id, self.state = self.update_state()
        return self.state

    def process_data(self, dataset_df, ta_period, price_idx):
        """
        Process the dataset by sampling, generating technical indicators and scaling.

        Args:
            dataset_df (pd.DataFrame): The input dataset as a pandas DataFrame.
            ta_period (int): The period used for calculating technical indicators.
            price_idx (int): The index of the price feature in the `features` list.

        Returns:
            dataset (np.ndarray): The processed and scaled dataset as a NumPy array.
            price_data (np.ndarray): The price data extracted from the dataset.
        """
        sampled_dfs = []

        # Loop through the range(sampling_period) to generate sampled datasets
        for i in range(self.sampling_period):
            # Sample the dataset
            sampled_df = self.resample(
                dataset_df[i::], str(self.sampling_period) + 'T')
            # Generate TAs for the sampled dataset
            sampled_df = self.generate_technical_indicators(
                sampled_df)
            # Append the sampled dataset to the list of sampled datasets
            sampled_dfs.append(sampled_df)

        # Concatenate the sampled datasets to create the final processed dataset
        processed_dataset_df = pd.concat(sampled_dfs)

        # Sort the final dataset to get the correct order
        processed_dataset_df = processed_dataset_df.sort_index()

        # Drop the first 5*ta_period*sample_period rows and the NaN values
        num_drop = 5*ta_period*self.sampling_period
        processed_dataset_df = processed_dataset_df.iloc[num_drop:]
        processed_dataset_df = processed_dataset_df.dropna()

        # Convert the dataset_df to a NumPy array and get the price data
        dataset = processed_dataset_df[self.features].to_numpy().astype(
            np.float32)
        price_data = dataset[:, price_idx]

        # Scale the dataset
        if not self.scaler:
            self.scaler = RobustScaler()
            dataset_scaled = self.scaler.fit_transform(
                dataset).astype(np.float32)
        else:
            dataset_scaled = self.scaler.transform(dataset).astype(np.float32)
        return dataset_scaled, price_data

    def resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Resample the dataset.

        Args:
            df (pd.DataFrame): The input dataset as a pandas DataFrame.
            interval (str): The sampling interval.

        Returns:
            df (pd.DataFrame): The resampled dataset as a pandas DataFrame.
        """
        d = {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        return df.resample(interval, origin="start").agg(d)

    def generate_technical_indicators(self, dataset_df, ta_period):
        """
        Generate technical indicators from the dataset.

        Args:
            dataset_df (pd.DataFrame): The input dataset as a pandas DataFrame.
            ta_period (int): The period used for calculating technical indicators.

        Returns:
            dataset_df (pd.DataFrame): The dataset with the technical indicators.
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
        return dataset_df

    def step(self, action):
        """
        Perform one step in the environment.

        Args:
            action (float): The chosen action.

        Returns:
            state (np.ndarray): The current state.
            reward (float): The reward obtained from the action.
            done (bool): Indicates whether the episode is finished.
        """
        if self.done:
            return self.state, self.reward, self.done
        self.reward = self.calculate_reward(action)
        self.balance += self.reward
        self.state_id, self.state = self.update_state()
        self.step_count += 1
        max_steps = self.episode_length if not self.eval_episode else self.eval_episode_length
        if (self.step_count >= max_steps) or (self.balance <= 0):
            self.done = True
            if not self.eval_episode:
                self.prev_episode_lengths.append(self.step_count)
                log_str = "Episode length: {}/{}, Losing actions: {}, Neutral actions: {}, Winning actions: {}, Mean Episode Length (25): {}".format(
                    self.step_count, max_steps, self.action_results_count[0], self.action_results_count[1], self.action_results_count[2], np.mean(self.prev_episode_lengths))
            else:
                log_str = "=== Evaluation Episode length: {}/{}, Losing actions: {}, Neutral actions: {}, Winning actions: {} ===".format(
                    self.step_count, max_steps, self.action_results_count[0], self.action_results_count[1], self.action_results_count[2])
            if self.logging:
                self.log(log_str)
            if self.verbose > 1:
                print(log_str)
        return self.state, self.reward, self.done

    def calculate_reward(self, action):
        """
        Calculate the reward based on the agent's action and the winning action.

        Args:
            action (float): The chosen action.

        Returns:
            reward (float): The reward obtained from the action.
        """
        if action not in self.action_space:
            raise ValueError('Invalid action!')
        else:
            self.action = action
        prediction_id = self.state_id + self.prediction_period
        self.current_price = self.price_data[self.state_id]
        self.prediction_price = self.price_data[prediction_id]
        self.winning_action = np.sign(
            self.prediction_price - self.current_price)
        if self.action*self.winning_action < 0:
            reward = self.reward_space[0]
            self.action_results_count[0] += 1
        elif self.action*self.winning_action == 0:
            reward = self.reward_space[1]
            self.action_results_count[1] += 1
        elif self.action*self.winning_action > 0:
            reward = self.reward_space[2]
            self.action_results_count[2] += 1
        return reward

    def update_state(self):
        """
        Update the current state by randomly selecting a state within the data range.

        Returns:
            state_id (int): The index of the selected state.
            state (np.ndarray): The state corresponding to the selected state_id.
        """
        state_id = random.randrange(
            self.window_size - 1, self.num_data - self.prediction_period)
        start = state_id - self.window_size + 1
        end = state_id + 1
        state = self.dataset[start:end, :]
        return state_id, state

    def get_action_space(self):
        """
        Return the action space.

        Returns:
            action_space (list): The available actions.
        """
        return self.action_space

    def get_scaler(self):
        """
        Return the scaler.

        Returns:
            scaler (sklearn.preprocessing.RobustScaler): The scaler.
        """
        return self.scaler

    def log(self, line):
        """
        Log a line to the log file.

        Args:
            line (str): The line to log to the log file.
        """
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
