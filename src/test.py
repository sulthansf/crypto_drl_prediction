import os
import time
import joblib
from env import PredictionGameEnvironment
from agent import PredictionGameDRLAgent


def main():
    # Set the inputs
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_name = 'BTCUSDT_3000_days_5_min'
    scaler_name = 'scaler_20230730_131839'
    q_network_name = 'q_network_20230730_131839_reward_52_0'
    dataset_path = os.path.join(
        project_path, 'datasets', dataset_name + '.bin')
    scaler_path = os.path.join(project_path, 'scalers', scaler_name + '.bin')
    q_network_path = os.path.join(
        project_path, 'models', q_network_name + '.keras')
    dataset_df = joblib.load(dataset_path)
    features = ['open', 'high', 'low', 'close', 'volume', 'bb_upper',
                'bb_middle', 'bb_lower', 'macd', 'signal', 'rsi',  'stoch_k', 'stoch_d']
    sampling_interval = 5
    resampling_interval = 5
    prediction_interval = 30
    state_increment = 1
    ta_period = 14
    window_size = 36
    episode_length = 1000
    test_episode_length = 50000
    state_shape = (window_size, len(features))
    timestr = time.strftime("%Y%m%d_%H%M%S")
    agent_log_path = os.path.join(
        project_path, 'log', 'agent_log_' + timestr + '.txt')
    env_log_path = os.path.join(
        project_path, 'log', 'env_log_' + timestr + '.txt')

    # Create the environment
    env = PredictionGameEnvironment(dataset_df, features, sampling_interval, resampling_interval, prediction_interval, state_increment,
                                    ta_period, window_size, episode_length, test_episode_length, scaler_path=scaler_path, verbose=2, logging=True, log_path=env_log_path)
    action_space = env.get_action_space()

    # Create the agent
    agent = PredictionGameDRLAgent(
        state_shape, action_space, verbose=2, logging=True, log_path=agent_log_path, auto_save=False)

    # Load q_network
    agent.load_q_network(q_network_path)

    # Test the agent on the environment
    episodes = 100
    agent.test(env, episodes, random_state=False)


if __name__ == '__main__':
    main()
