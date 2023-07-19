import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from finta import TA


class PredictionGameEnvironment:
    """
    A prediction game environment for the RL agent.
    """
    
    def __init__(self, dataset_df, features, ta_period, window_size, episode_length, prediction_period):
        self.features = features
        price_idx = self.features.index('close')
        self.dataset, self.price_data = self.process_data(
            dataset_df, ta_period, price_idx)
        self.num_data = len(self.dataset)
        self.window_size = window_size
        self.episode_length = episode_length
        self.prediction_period = prediction_period
        self.action_space = [-1.0, 0.0, 1.0]
        self.reward_space = [-1.0, 0.0, 0.7]
        self.reset()

    def reset(self):
        """
        Reset the environment for a new episode and return the initial state.

        Returns:
            state (np.ndarray): The initial state of the environment.
        """
        self.done = False
        self.step_count = 0
        self.cumulative_reward = 10.0
        self.reward = None
        self.action = None
        self.winning_action = None
        self.state_id, self.state = self.update_state()
        return self.state

    def process_data(self, dataset_df, ta_period, price_idx):
        """
        Process the dataset by calculating technical indicators and scaling the data.

        Args:
            dataset_df (pd.DataFrame): The input dataset as a pandas DataFrame.
            ta_period (int): The period used for calculating technical indicators.
            price_idx (int): The index of the price feature in the `features` list.

        Returns:
            dataset (np.ndarray): The processed and scaled dataset as a NumPy array.
            price_data (np.ndarray): The price data extracted from the dataset.
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
        dataset_df = dataset_df.dropna()
        num_drop = 5*ta_period
        dataset_df = dataset_df.iloc[num_drop:]
        dataset = dataset_df[self.features].to_numpy()
        price_data = dataset[:, price_idx]
        dataset_scaled = RobustScaler().fit_transform(dataset)
        return dataset_scaled, price_data

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
        self.cumulative_reward += self.reward
        self.state_id, self.state = self.update_state()
        self.step_count += 1
        if (self.step_count >= self.episode_length) or (self.cumulative_reward <= 0):
            self.done = True
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
        current_price = self.price_data[self.state_id]
        prediction_price = self.price_data[prediction_id]
        self.winning_action = np.sign(prediction_price - current_price)
        if self.action*self.winning_action < 0:
            reward = self.reward_space[0]
        elif self.action*self.winning_action == 0:
            reward = self.reward_space[1]
        elif self.action*self.winning_action > 0:
            reward = self.reward_space[2]
        return reward

    def update_state(self):
        """
        Update the current state by randomly selecting a state within the data range.

        Returns:
            state_id (int): The index of the selected state.
            state (np.ndarray): The state corresponding to the selected state_id.
        """
        state_id = random.randrange(
            self.window_size - 1, self.num_data)
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
