import time
import argparse
import json
from player import PredictionGamePlayer


def main():
    """
    The main function.
    """
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='The config file name.')
    args = parser.parse_args()

    # Set the api key and secret
    api_key = 'YOUR_API_KEY'
    api_secret = 'YOUR_API_SECRET'

    # Set the config file
    config_file_name = args.config
    config_path = '../config/' + config_file_name

    # Read the json file
    config = json.load(open(config_path))

    # Set the inputs
    symbol = config['symbol']
    interval = config['interval']
    features = config['features']
    ta_period = config['ta_period']
    window_size = config['window_size']
    prediction_period = config['prediction_period']
    action_space = config['action_space']

    # Model and scaler paths
    model_name = config['model_name']
    scaler_name = config['scaler_name']
    model_path = '../models/' + model_name
    scaler_path = '../scalers/' + scaler_name

    # Create the player
    player = PredictionGamePlayer(api_key, api_secret, symbol, interval, features, ta_period, window_size,
                                  prediction_period, action_space, model_path, scaler_path, verbose=2, logging=True)

    # Play the game
    # player.next_action()


if __name__ == '__main__':
    main()
