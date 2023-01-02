import os
import tensorflow as tf
from dddqn_tf2.dueling_dqn_network import DDDQN
from dddqn_tf2.memory import ReplayBuffer
import numpy as np

class Agent(object):
    def __init__(self,
                 lr,
                 gamma,
                 n_actions,
                 epsilon,
                 batch_size,
                 input_dims,
                 epsilon_dec=1e-3,
                 epsilon_end=0.01,
                 mem_size=1000000,
                 replace=100):

        self.n_actions = n_actions
        self.gamma = gamma
        self.replace = replace
        self.trainstep = 0
        self.batch_size = batch_size

        # Epsilon information
        self.epsilon = epsilon
        self.min_epsilon = epsilon_end
        self.epsilon_decay = epsilon_dec

        # Memory information
        self.memory = ReplayBuffer(mem_size, input_dims)

        # DQN
        self.q_net = DDDQN(n_actions)
        self.target_net = DDDQN(n_actions)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)


    def add_experience(self, state, action, reward, new_state, done):
        self.memory.add_experience(state, action, reward, new_state, done)

    def act(self, state):

        # With chance epsilon, take a random action
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(self.n_actions)])

        else:
            # Otherwise, query the DQN for an action
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            return action

    def update_target_network(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        # epsilon = explore_probability
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def learn(self):
        """
                Sample a batch and use it to improve the DQN
        """

        if self.memory.mem_cntr < self.batch_size:
            return

        if self.trainstep % self.replace == 0:
            self.update_target_network()
            print("Target Model updated")

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)

        # Main DQN estimates best action in new states
        target = self.q_net.predict(states)

        # Target DQN estimates q-vals for new states
        next_state_val = self.target_net.predict(next_states)

        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Calculate targets (bellman equation)
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones

        self.q_net.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1

    def save_model(self, folder_name, **kwargs):
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

            # Save DQN and target DQN
            self.q_net.save(folder_name + "/model.h5")
            self.target_net.save(folder_name + "/target_model.h5")

            # Save replay buffer
            # self.memory.save(folder_name + '/replay-buffer')

    def load_model(self, folder_name, load_replay_buffer=True):

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.q_net = tf.keras.models.load_model(folder_name + "/model.h5")
        self.target_net = tf.keras.models.load_model(folder_name + "/model.h5")
        # self.optimizer = self.DQN.optimizer

        # Load replay buffer
        # if load_replay_buffer:
        #     self.replay_buffer.load(folder_name + '/replay-buffer')