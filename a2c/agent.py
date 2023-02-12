import pylab
import cv2
import numpy as np
from a2c.networks import A2CModel
import os
from tensorflow.keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class A2CAgent:
    # Actor-Critic Main Optimization Algorithm
    def __init__(self, env_name, env, state_size, num_actions):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.env = env

        self.action_size = num_actions
        self.EPISODES, self.max_average = 300, 0.0  # specific for pong
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
        state = np.expand_dims(state, axis=0)
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

    def learn(self):

        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)

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


    def train(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            done, score, SAVING = False, 0, ''
            while not done:
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
                    self.learn()

        # close environemnt when finish training
        self.env.game.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        game = self.env.game
        for e in range(100):

            self.env.game.new_episode()
            state = game.get_state().screen_buffer
            state = self.env.stack_frames(self.env.stacked_frames, state, True)
            state = np.reshape(state, [1, *self.state_size])

            while not game.is_episode_finished():
                Qs = self.Actor.predict(state)

                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = self.env.possible_actions[int(choice)]

                game.make_action(action)
                done = game.is_episode_finished()

                if done:
                    # print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
                else:
                    next_state = game.get_state().screen_buffer
                    next_state = self.env.stack_frames(self.env.stacked_frames, next_state, False)
                    next_state = np.reshape(next_state, [1, *self.state_size])
                    state = next_state

            score = game.get_total_reward()
            print("episode: {}/{}, score: {}".format(e, 100, score))
        # self.env.close()