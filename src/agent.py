import gc
import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque


class PredictionGameDRLAgent:
    def __init__(self, state_shape, action_space, epsilon_initial=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99, update_frequency=10, verbose=1, logging=False, log_path=None, auto_save=True, save_path=None, save_frequency=500):
        """
        Initialize the DRL agent.

        Args:
            state_shape (tuple): The shape of the input state.
            action_space (list): The list of possible actions.
            epsilon_initial (float): The initial value of epsilon.
            epsilon_decay (float): The decay rate of epsilon.
            epsilon_min (float): The minimum value of epsilon.
            gamma (float): The discount factor.
            update_frequency (int): The number of steps between each Q-network update.
            verbose (int): The verbosity level.
            logging (bool): Whether to log the training process or not.
            log_path (str): The path to the log file.
            auto_save (bool): Whether to automatically save the Q-network or not.
            save_path (str): The path to save the Q-network to.
            save_frequency (int): The number of episodes between each save.
        """
        # Set the inputs
        self.state_shape = state_shape
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.epsilon = epsilon_initial
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.verbose = verbose
        self.logging = logging
        self.auto_save = auto_save
        self.save_frequency = save_frequency
        self.project_path = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        timestr = time.strftime("%Y%m%d_%H%M%S")
        if log_path:
            self.log_path = log_path
        else:
            self.log_path = os.path.join(
                self.project_path, 'log', 'agent_log_' + timestr + '.txt')
        if save_path:
            self.save_path = save_path
        else:
            self.save_path = os.path.join(
                self.project_path, 'models', 'q_network_' + timestr + '.keras')

        # Create the Q-network and target Q-network
        self.q_network = self.create_q_network()
        self.target_q_network = self.create_q_network()
        self.update_target_q_network()

    def create_q_network(self):
        """
        Create the Q-network for the DRL agent.

        Args:
            input_shape (tuple): The shape of the input state.
            num_actions (int): The number of possible actions.

        Returns:
            q_network (tf.keras.Model): The Q-network model.
        """
        model = tf.keras.Sequential()

        # Convolutional Layers
        model.add(layers.Conv1D(filters=32, kernel_size=3,
                  input_shape=self.state_shape))
        model.add(layers.Activation('leaky_relu'))
        model.add(layers.Conv1D(filters=64, kernel_size=3))
        model.add(layers.Activation('leaky_relu'))
        model.add(layers.MaxPooling1D(pool_size=2))

        # Recurrent Layer (LSTM)
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(64))

        # Dense Layers
        model.add(layers.Dense(64))
        model.add(layers.Activation('leaky_relu'))
        model.add(layers.Dense(self.num_actions))

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0003), loss='mse')

        # Print the model summary
        model.summary()

        return model

    def update_target_q_network(self):
        """
        Update the target Q-network with the weights of the Q-network.
        """
        self.target_q_network.set_weights(self.q_network.get_weights())

    def choose_action(self, state, exploration=True):
        """
        Choose an action based on the current state.

        Args:
            state (np.ndarray): The current state.
            exploration (bool): Whether to perform exploration (choose random actions) or not.

        Returns:
            action (float): The chosen action.
        """
        if exploration and np.random.rand() <= self.epsilon:
            return self.action_space[random.choice(range(self.num_actions))]
        else:
            return self.action_space[np.argmax(self.q_network.predict_on_batch(np.array([state]))[0])]

    def train(self, env, episodes, batch_size, eval_frequency=10, random_state=False):
        """
        Train the DRL agent on the environment.

        Args:
            env (PredictionGameEnvironment): The environment to train the agent on.
            episodes (int): The number of episodes to train the agent.
            batch_size (int): The batch size for training the agent.
            eval_frequency (int): The number of episodes between each evaluation.
        """
        replay_buffer = deque(maxlen=10000)
        steps_since_update = 0
        evaluation_rewards = deque(maxlen=10)
        highest_mean_evaluation_reward = 0

        # Train the agent on the environment
        for episode in range(episodes):
            # Reset the environment
            state = env.reset()
            done = False
            total_reward = 0

            # Collect the garbage
            gc.collect()

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action, random_state)
                total_reward += reward

                replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                steps_since_update += 1

                if (len(replay_buffer) >= batch_size) and (steps_since_update >= self.update_frequency):
                    self.train_q_network(replay_buffer, batch_size)
                    self.update_target_q_network()
                    steps_since_update = 0

            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)

            log_str = "Episode: {}/{} | Episode Reward: {}".format(
                episode+1, episodes, total_reward)
            if self.logging:
                self.log(log_str)
            if self.verbose > 1:
                print(log_str)

            # Save the Q-network every save_frequency episodes
            if self.auto_save and (episode+1) % self.save_frequency == 0:
                save_file, save_extension = os.path.splitext(
                    self.save_path)
                save_path = "{}_episode_{}{}".format(
                    save_file, episode+1, save_extension)
                self.save_q_network(save_path)
                log_str = "=== Saved Q-Network at Episode: {} ===".format(
                    episode+1)
                if self.logging:
                    self.log(log_str)
                if self.verbose > 0:
                    print(log_str)

            # Evaluate the agent every eval_frequency episodes
            if (episode + 1) % eval_frequency == 0:
                evaluation_reward = self.evaluate(env)
                log_str = "=== Evaluation Reward: {} ===".format(
                    evaluation_reward)
                if self.logging:
                    self.log(log_str)
                if self.verbose > 0:
                    print(log_str)

                # Save the Q-network if the mean evaluation reward is higher
                evaluation_rewards.append(evaluation_reward)
                if np.mean(evaluation_rewards) > highest_mean_evaluation_reward:
                    highest_mean_evaluation_reward = np.mean(
                        evaluation_rewards)
                    if self.auto_save:
                        save_file, save_extension = os.path.splitext(
                            self.save_path)
                        save_path = "{}_reward_{}{}".format(
                            save_file, int(highest_mean_evaluation_reward), save_extension)
                        self.save_q_network(save_path)
                        log_str = "=== Saved Q-Network with Mean (10) Evaluation Reward: {} ===".format(
                            highest_mean_evaluation_reward)
                        if self.logging:
                            self.log(log_str)
                        if self.verbose > 0:
                            print(log_str)

    def train_q_network(self, replay_buffer, batch_size):
        """
        Train the Q-network on a batch of data from the replay buffer.

        Args:
            replay_buffer (deque): The replay buffer.
            batch_size (int): The batch size for training the agent.
        """
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        current_q = self.q_network.predict_on_batch(states)
        next_q = self.target_q_network.predict_on_batch(next_states)

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q[i])

            current_q[i][self.action_space.index(actions[i])] = target

        self.q_network.train_on_batch(states, current_q)

    def evaluate(self, env, random_state=False):
        """
        Evaluate the DRL agent on the environment for one episode.

        Args:
            env (PredictionGameEnvironment): The environment to evaluate the agent on.

        Returns:
            evaluation_reward (float): The total reward obtained during the evaluation episode.
        """
        # Reset the environment
        state = env.reset(eval=True)
        done = False
        evaluation_reward = 0

        while not done:
            action = self.choose_action(state, exploration=False)
            next_state, reward, done = env.step(action, random_state)
            evaluation_reward += reward
            state = next_state

        return evaluation_reward

    def test(self, env, episodes, random_state=False):
        """
        Test the DRL agent on the environment for multiple episodes.

        Args:
            env (PredictionGameEnvironment): The environment to test the agent on.
            episodes (int): The number of episodes to test the agent.
            random_state (bool): Whether to use random states or not.

        Returns:
            test_rewards (list): The total rewards obtained during the test episodes.
        """
        # Test the agent on the environment
        for episode in range(episodes):
            # Collect the garbage
            gc.collect()
            # Evaluate the agent
            test_reward = self.evaluate(env, random_state)
            log_str = "Test Episode: {}/{} | Episode Reward: {}".format(
                episode+1, episodes, test_reward)
            if self.logging:
                self.log(log_str)
            if self.verbose > 1:
                print(log_str)

    def save_q_network(self, path):
        """
        Save the DRL agent.

        Args:
            path (str): The path to save the agent to.
        """
        self.q_network.save(path)

    def load_q_network(self, path):
        """
        Load the DRL agent.

        Args:
            path (str): The path to load the agent from.
        """
        self.q_network = tf.keras.models.load_model(path)
        self.target_q_network = tf.keras.models.load_model(path)

    def log(self, line):
        """
        Log a line to the log file.

        Args:
            line (str): The line to log to the log file.
        """
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
