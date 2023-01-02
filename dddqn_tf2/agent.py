import tensorflow as tf
from dddqn_tf2.model import DDDQN
from dddqn_tf2.memory import ReplayBuffer
import numpy as np
from tensorflow.keras.models import load_model

class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000, replace=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = epsilon_end
        self.epsilon_decay = epsilon_dec
        self.replace = replace
        self.trainstep = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.batch_size = batch_size
        self.q_net = DDDQN(n_actions)
        self.target_net = DDDQN(n_actions)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)
        self.n_actions = n_actions

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice([i for i in range(self.n_actions)])

        else:
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            return action

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        # epsilon = explore_probability
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.trainstep % self.replace == 0:
            self.update_target()
            print("Target Model updated")

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)

        max_action = np.argmax(self.q_net.predict(next_states), axis=1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones
        self.q_net.train_on_batch(states, q_target)
        self.update_epsilon()
        self.trainstep += 1

    def save_model(self):
        self.q_net.save("saved_model/model.h5")
        self.target_net.save("saved_model/target_model.h5")

    def load_model(self):
        self.q_net = load_model("saved_model/model.h5")
        self.target_net = load_model("saved_model/model.h5")