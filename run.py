import time
import joblib
from env import PredictionGameEnvironment
from agent import PredictionGameDRLAgent


def main():

    # Set the inputs
    dataset_df = joblib.load('ETH_1000_days_5_min.bin')
    features = ['open', 'high', 'low', 'close', 'volume', 'bb_upper',
                'bb_middle', 'bb_lower', 'macd', 'signal', 'rsi',  'stoch_k', 'stoch_d']
    ta_period = 12
    window_size = 12
    episode_length = 1000
    prediction_period = 1
    state_shape = (window_size, len(features))
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Create the environment
    env = PredictionGameEnvironment(
        dataset_df, features, ta_period, window_size, episode_length, prediction_period, logging=True)
    action_space = env.get_action_space()

    # Create the agent
    agent = PredictionGameDRLAgent(state_shape, action_space, epsilon_initial=1.0,
                                   epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99, update_frequency=10, logging=True)

    # Save scaler
    joblib.dump(env.get_scaler(), 'scaler_drl_' +
                timestr + '.bin', compress=True)

    # Train the agent on the environment
    episodes = 1000
    batch_size = 64
    agent.train(env, episodes, batch_size)

    # Save the agent q-network
    agent.save_q_network('q_network_drl_' + timestr + '.h5')


if __name__ == '__main__':
    main()
