import time
import joblib
from env import PredictionGameEnvironment
from agent import PredictionGameDRLAgent


def main():
    # Set the inputs
    dataset_df = joblib.load('../datasets/BTCUSDT_3000_days_5_min.bin')
    features = ['open', 'high', 'low', 'close', 'volume', 'bb_upper',
                'bb_middle', 'bb_lower', 'macd', 'signal', 'rsi',  'stoch_k', 'stoch_d']
    sampling_interval = 5
    resampling_interval = 5
    prediction_interval = 30
    ta_period = 14
    window_size = 36
    episode_length = 1000
    eval_episode_length = 10000
    state_shape = (window_size, len(features))
    timestr = time.strftime("%Y%m%d_%H%M%S")
    agent_log_path = '../log/agent_log_' + timestr + '.txt'
    env_log_path = '../log/env_log_' + timestr + '.txt'
    scaler_save_path = '../scalers/scaler_' + timestr + '.bin'
    q_network_save_path = '../models/q_network_' + timestr + '.keras'

    # Create the environment
    env = PredictionGameEnvironment(dataset_df, features, sampling_interval, resampling_interval, prediction_interval, ta_period, window_size,
                                    episode_length, eval_episode_length, scaler_path=None, verbose=2, logging=True, log_path=env_log_path)
    action_space = env.get_action_space()

    # Create the agent
    agent = PredictionGameDRLAgent(state_shape, action_space, epsilon_initial=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99,
                                   update_frequency=32, verbose=2, logging=True, log_path=agent_log_path, auto_save=True, save_path=q_network_save_path)

    # Save scaler
    joblib.dump(env.get_scaler(), scaler_save_path, compress=True)

    # Train the agent on the environment
    episodes = 1000
    batch_size = 64
    agent.train(env, episodes, batch_size)


if __name__ == '__main__':
    main()
