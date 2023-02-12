import os
import sys
import random
import pylab
import numpy as np
import tensorflow as tf
from collections import deque

from dddqn_tf2_v2.network import DQNModel
from dddqn_tf2_v2.memory import Memory
from utils.utils import plotLearning


class Agent(object):
    def __init__(self,
                 env,
                 lr,
                 gamma,
                 n_actions,
                 epsilon,
                 batch_size,
                 state_size,
                 action_shape,
                 epsilon_dec,
                 epsilon_end,
                 mem_size,
                 total_episodes,
                 max_steps,
                 pretrain_length,
                 writer):
        #Environment and parameters
        self.env = env
        self.env_name = "VizDoom"
        self.action_size = n_actions
        self.action_shape = action_shape
        self.max_average = 0
        self.best_reward = 0
        self.pretrain_length = pretrain_length
        self.EPISODES = total_episodes
        self.max_steps = max_steps
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

        # Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.misc, self.prev_misc = [], []

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
        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size[0],self.state_size[1],self.state_size[2]))
        next_state = np.zeros((self.batch_size, self.state_size[0],self.state_size[1],self.state_size[2]))
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

    def shape_reward(self, reward, misc, prev_misc):
        if self.env.game.get_state():
            # Check any kill count
            if (misc[0] > prev_misc[0]):  # KILLCOUNT
                reward = reward + 1

            if (misc[1] < prev_misc[1]):  # Use ammo
                reward = reward - 0.1

            if (misc[2] < prev_misc[2]):  # Loss HEALTH
                reward = reward - 0.1

        return reward

    def train(self):

        print("Filling memory: ", self.pretrain_length)
        for i in range(self.pretrain_length):
            state = self.env.reset()
            sys.stdout.write(f"\r{str((i / self.pretrain_length) * 100)} %")
            done = False

            misc = self.env.get_variables()
            prev_misc = misc

            while not done:
                # Random action
                action = random.choice(self.action_shape)
                next_state, reward, done, _ = self.env.step(action)
                reward = self.shape_reward(reward, misc, prev_misc)
                self.remember(state, action, reward, next_state, done)
                prev_misc = misc
                if done:
                    break;
        print('\nDone initializing memory')

        decay_step = 0

        # Init the game

        scores = []
        eps_history = []

        # Update the parameters of our TargetNetwork with DQN_weights

        for episode in range(self.EPISODES):
            eps_reward = []
            total_reward = 0.0
            step = 0

            # Initialize the rewards of the episode
            eps_history.append(self.epsilon)

            print('Starting episode: ', episode, ', epsilon: %.4f' % self.epsilon + ' ')
            # Start a new episode
            state = self.env.reset()

            misc = self.env.get_variables()
            prev_misc = misc

            while step < self.max_steps:
                step += 1
                decay_step += 1

                action, explore_probability = self.act(state, decay_step, episode)

                # Do the action
                next_state, reward, done, _ = self.env.step(action)
                reward = self.shape_reward(reward, misc, prev_misc)

                # Memoorize
                self.remember(state, action, reward, next_state, done)

                prev_misc = misc
                # Add the reward to total reward
                total_reward += reward
                eps_reward.append(reward)
                state = next_state

                if done:
                    if total_reward > self.best_reward:
                        self.best_reward = total_reward
                    # the episode ends so no next state
                    # every step update target model
                    self.update_target_model()

                    # Get the total reward of the episode
                    total_reward = np.sum(total_reward)

                    # every episode, plot the result
                    average = self.PlotModel(total_reward, episode)

                    self.learn()
                    step = self.max_steps

            scores.append(total_reward)
            print('Episode: {}/{}, Total reward: {}, e: {:.2}, average: {} {}'.format(episode, self.EPISODES,
                                                                                      total_reward,
                                                                                      self.epsilon, np.mean(eps_reward),
                                                                                      "SAVING"))
            self.plotLearning(episode, scores, eps_history)
            self.writer.add_scalar("main/ep_reward", total_reward, episode)
            self.writer.add_scalar("main/mean_ep_reward", np.mean(eps_reward), episode)
            self.writer.add_scalar("main/max_ep_reward", self.best_reward, episode)

            # Save model every 5 episodes
            if episode % 5 == 0:
                self.save_model("Models")
                print("Model Saved")

        self.env.game.close()
        self.writer.close()
        # tensorboard --logdir=logs/
        # http://localhost:6006/

    def test(self):
        self.load_model("Models")
        game = self.env.game

        for i in range(100):

            game.new_episode()
            state = game.get_state().screen_buffer
            state = self.env.stack_frames(state, True)
            state = np.reshape(state, [1, *self.state_size])

            while not game.is_episode_finished():
                # Estimate the Qs values state
                Qs = self.model.predict(state)

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = self.env.possible_actions[int(choice)]

                game.make_action(action)
                done = game.is_episode_finished()

                if done:
                    break

                else:
                    next_state = game.get_state().screen_buffer
                    next_state = self.env.stack_frames(next_state, False)
                    next_state = np.reshape(next_state, [1, *self.state_size])
                    state = next_state

            score = game.get_total_reward()
            print("Score: ", score)

        game.close()

    def save_model(self, folder_name, **kwargs):
        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.model.save(folder_name + "/model.h5")
        self.target_model.save(folder_name + "/target_model.h5")

    def load_model(self, folder_name):

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