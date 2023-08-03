import time
import joblib
from env import PredictionGameEnvironment
from agent import PredictionGameDRLAgent


def main():
    # Set the inputs
    scaler_name = 'scaler_20210320_162000'
    q_network_name = 'q_network_20210320_162000'
    scaler_path = '../scalers/scaler_' + scaler_name + '.bin'
    q_network_path = '../models/q_network_' + q_network_name + '.keras'
    dataset_df = joblib.load('../datasets/ETHUSDT_1000_days_5_min.bin')
    features = ['open', 'high', 'low', 'close', 'volume', 'bb_upper',
                'bb_middle', 'bb_lower', 'macd', 'signal', 'rsi',  'stoch_k', 'stoch_d']
    sampling_period = 1
    ta_period = 12
    window_size = 12
    episode_length = 1000
    test_episode_length = 50000
    prediction_period = 1
    state_shape = (window_size, len(features))
    timestr = time.strftime("%Y%m%d_%H%M%S")
    agent_log_path = '../log/agent_log_' + timestr + '.txt'
    env_log_path = '../log/env_log_' + timestr + '.txt'

    # Create the environment
    env = PredictionGameEnvironment(dataset_df, features, sampling_period, ta_period, window_size, episode_length,
                                    test_episode_length, prediction_period, scaler_path=scaler_path, verbose=2, logging=True, log_path=env_log_path)
    action_space = env.get_action_space()

    # Create the agent
    agent = PredictionGameDRLAgent(
        state_shape, action_space, verbose=2, logging=True, log_path=agent_log_path, auto_save=False)

    # Load q_network
    agent.load_q_network(q_network_path)

    # Test the agent on the environment
    episodes = 100
    agent.test(env, episodes)


if __name__ == '__main__':
    main()
