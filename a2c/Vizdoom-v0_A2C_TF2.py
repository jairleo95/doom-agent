#https://pylessons.com/A2C-reinforcement-learning
#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
#https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py

#Advantage Actor-Critic implementation

import os

from wrappers.doom_wrapper import DoomEnv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pylab
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import RMSprop
import cv2


def A2CModel(input_shape, action_space, lr):
    #Advantage Actor-Critic
    X_input = Input(input_shape)
    print("input_shape: ", input_shape)

    X = Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")(X_input)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(128, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Flatten()(X)

    # X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs=X_input, outputs=action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs=X_input, outputs=value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    Actor.summary()
    Critic.summary()

    return Actor, Critic


class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name, env, state_size, num_actions):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.env = env

        self.action_size = num_actions
        self.EPISODES, self.max_average = 50, 0.0  # specific for pong
        self.lr = 0.000025

        # Instantiate games and plot memory
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = state_size

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Create Actor-Critic network model
        self.Actor, self.Critic = A2CModel(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)

    def remember(self, state, action, reward):
        # store episode actions to memory
        state = np.expand_dims(state, axis=0)
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        print('act.state.shape', state.shape)
        state = np.expand_dims(state, axis=0)
        print('act.state.shape.expand_dims', state.shape)
        #act.state.shape.expand_dims (1, 4, 64, 64)
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates
    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.99  # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        print("discount_rewards.reward.length", len(reward))
        for i in reversed(range(0, len(reward))):
            if reward[i] != 0:  # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= np.std(discounted_r)  # divide by standard deviation
        return discounted_r

    def replay(self):
        episode_length = len(self.states)

        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        print("replay.states.shape", states.shape)
        print("replay.actions.shape", actions.shape)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(self.rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []

    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        # self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        # self.Critic.save(self.Model_name + '_Critic.h5')

    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        # if str(episode)[-2:] == "00":  # much faster than episode % 100
        if episode % 5:
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path + ".png")
            except OSError:
                pass

        return self.average[-1]

    def imshow(self, image, rem_step=0):
        cv2.imshow(self.Model_name + str(rem_step), image[rem_step, ...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return


    def run(self):
        max_steps = 100
        for e in range(self.EPISODES):
            step = 0
            state = self.env.reset()
            done, score, SAVING = False, 0, ''
            while step < max_steps:
                step += 1
                # self.env.render()
                # Actor picks an action
                action = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)
                # Memorize (state, action, reward) for training
                self.remember(state, action, reward)
                # Update current state
                state = next_state
                score += reward
                if done:
                    average = self.PlotModel(score, e)
                    # saving best models
                    if average >= self.max_average:
                        self.max_average = average
                        self.save()
                        SAVING = "SAVING"
                    else:
                        SAVING = ""
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average,
                                                                                 SAVING))
                    self.replay()
                    step = max_steps

        # close environemnt when finish training
        self.env.game.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):

            env.game.new_episode()
            state = env.game.get_state().screen_buffer
            state = env.stack_frames(env.stacked_frames, state, True)
            state = np.reshape(state, [1, *agent.state_size])

            while not env.game.is_episode_finished():
                Qs = self.Actor.predict(state)

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = env.possible_actions[int(choice)]

                env.game.make_action(action)
                done = env.game.is_episode_finished()

                if done:
                    # print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
                else:
                    next_state = env.game.get_state().screen_buffer
                    next_state = env.stack_frames(env.stacked_frames, next_state, False)
                    next_state = np.reshape(next_state, [1, *agent.state_size])
                    state = next_state

            score = env.game.get_total_reward()
            print("Score: ", score)
        # self.env.close()


if __name__ == "__main__":
    env_name = 'Vizdoom-v0'
    state_size = (4, 64, 64)
    env = DoomEnv(stack_size=4, img_shape=(64, 64))
    num_actions, action_shape = env.create_env()
    agent = A2CAgent(env_name, env, state_size, num_actions)
    # agent.run()
    agent.test('Models/Vizdoom-v0_A2C_2.5e-05_Actor.h5', '')
    # agent.test('PongDeterministic-v4_A2C_1e-05_Actor.h5', '')