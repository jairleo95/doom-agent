import os
import random
import pylab
from collections import deque
import tensorflow as tf
from dddqn_tf2.dueling_dqn_network import DQNModel
from dddqn_tf2.memory import Memory
from utils.utils import plotLearning
import numpy as np

class Agent(object):
    def __init__(self,
                 lr,
                 gamma,
                 n_actions,
                 epsilon,
                 batch_size,
                 state_size,
                 epsilon_dec,
                 epsilon_end,
                 mem_size,
                 total_episodes, writer):
        #Environment and parameters
        self.env_name = "VizDoom"
        self.action_size = n_actions
        self.max_average = 0
        self.best_reward = 0
        self.results_filename = 'results/' + str(total_episodes) + 'Games' + 'Gamma' + str(gamma) + 'Alpha' + str(
            lr) + 'Memory' + str(mem_size) + '_vizdoom_dddqn__tf2.png'

        self.gamma = gamma
        self.trainstep = 0

        self.batch_size = batch_size

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = epsilon  # exploration probability at start
        self.epsilon_min = epsilon_end  # minimum exploration probability
        self.epsilon_decay = epsilon_dec  # exponential decay rate for exploration prob

        # Memory information
        self.MEMORY = Memory(mem_size)
        self.memory = deque(maxlen=2000)

        # defining model parameters
        self.ddqn = True  # use doudle deep q network
        self.Soft_Update = False  # use soft parameter update
        self.dueling = True  # use dealing netowrk
        self.epsilon_greedy = False  # use epsilon greedy strategy
        self.USE_PER = False  # use priority experienced replay
        self.TAU = 0.1  # target network soft update hyperparameter

        self.scores, self.episodes, self.average = [], [], []

        # DQN
        self.state_size = state_size

        # create main model and target model
        self.model = DQNModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)
        self.target_model = DQNModel(input_shape=self.state_size, action_space=self.action_size, dueling=self.dueling)

        self.writer = writer

    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, new_state, done):
        experience = state, action, reward, new_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state, decay_step, episode):
        self.writer.add_scalar("DQL/epsilon", self.epsilon, episode)

        # EPSILON GREEDY STRATEGY
        if self.epsilon_greedy:
            # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        # OLD EPSILON STRATEGY
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return np.random.choice([i for i in range(self.action_size)]), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            actions = self.model.predict(np.array([state]))
            action = np.argmax(actions)
            return action, explore_probability

    def learn(self):
        """
                Sample a batch and use it to improve the DQN
        """
        # if len(self.memory) < self.train_start:
        #     return
        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, 84,84,4))
        next_state = np.zeros((self.batch_size, 84,84,4))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                if self.ddqn:  # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(action)] - target[indices, np.array(action)])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def save_model(self, folder_name, **kwargs):
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.model.save(folder_name + "/model.h5")
        self.target_model.save(folder_name + "/target_model.h5")

    def load_model(self, folder_name, load_replay_buffer=True):

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.model = tf.keras.models.load_model(folder_name + "/model.h5")
        self.target_model = tf.keras.models.load_model(folder_name + "/model.h5")

        # Load replay buffer

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn: dqn = 'DDQN_'
        if self.Soft_Update: softupdate = '_soft'
        if self.dueling: dueling = '_Dueling'
        if self.epsilon_greedy: greedy = '_Greedy'
        if self.USE_PER: PER = '_PER'
        try:
            pylab.savefig("results/"+dqn + self.env_name + softupdate + dueling + greedy + PER + ".png")
        except OSError:
            pass

        return self.average[-1]

    def plotLearning(self, total_episodes, scores, eps_history):

        x = [i for i in range(total_episodes+1)]
        plotLearning(x, scores, eps_history, self.results_filename)