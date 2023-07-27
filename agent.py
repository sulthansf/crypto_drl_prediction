import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque


class PredictionGameDRLAgent:
    def __init__(self, state_shape, action_space, epsilon_initial=1.0, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.99, update_frequency=10):
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
                                activation='relu', input_shape=self.state_shape))
        model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))

        # Recurrent Layer (LSTM)
        model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.LSTM(64))

        # Dense Layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.num_actions, activation='linear'))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

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
            return self.action_space[np.argmax(self.q_network.predict(np.array([state]), verbose=0))]

    def train(self, env, episodes, batch_size, eval_frequency=10):
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

        # Train the agent on the environment
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
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

            print("Episode: {}/{} | Total Reward: {}".format(
                episode+1, episodes, total_reward))

            # Evaluate the agent every eval_frequency episodes
            if episode % eval_frequency == 0:
                evaluation_reward = self.evaluate(env)
                print("=== Evaluation Reward: {} ===".format(evaluation_reward))

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

        current_q = self.q_network.predict(states, verbose=0)
        next_q = self.target_q_network.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q[i])

            current_q[i][self.action_space.index(actions[i])] = target

        self.q_network.fit(states, current_q, verbose=0)

    def evaluate(self, env):
        """
        Evaluate the DRL agent on the environment for one episode.

        Args:
            env (PredictionGameEnvironment): The environment to evaluate the agent on.

        Returns:
            evaluation_reward (float): The total reward obtained during the evaluation episode.
        """
        state = env.reset()
        done = False
        evaluation_reward = 0

        while not done:
            action = self.choose_action(state, exploration=False)
            next_state, reward, done = env.step(action)
            evaluation_reward += reward
            state = next_state

        return evaluation_reward
