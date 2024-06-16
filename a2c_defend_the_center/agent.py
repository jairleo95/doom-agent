import numpy as np
import pylab
import cv2
import os

from a2c_defend_the_center.networks import Networks

class A2CAgent:

    def __init__(self, env, state_size, action_size, statistics_file):
        # get size of state and action
        self.env_name = "Vizdoom_defend_the_center"
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.observe = 0
        self.frame_per_action = 4
        self.env = env
        self.EPISODES = 300

        # These are hyper parameters for the Policy Gradient
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001

        # Model for policy and critic network
        self.actor = Networks.actor_network(state_size, action_size, self.actor_lr)
        self.critic = Networks.critic_network(state_size, self.value_size, self.critic_lr)

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # Performance Statistics
        self.stats_window_size = 20  # window size for computing rolling statistics
        self.mavg_score = []  # Moving Average of Survival Time
        self.var_score = []  # Variance of Survival Time
        self.mavg_ammo_left = []  # Moving Average of Ammo used
        self.mavg_kill_counts = []  # Moving Average of Kill Counts

        self.statistics_file = statistics_file
        self.Save_Path = 'Models'

        self.scores, self.episodes, self.average = [], [], []
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format(self.env_name, self.actor_lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)


    # using the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        policy = self.actor.predict(state).flatten()
        # print("policy: ", policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def learn(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        # Standardized discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        if np.std(discounted_rewards):
            discounted_rewards /= np.std(discounted_rewards)
        else:
            self.states, self.actions, self.rewards = [], [], []
            print('std = 0!')
            return 0

        update_inputs = np.zeros(((episode_length,) + self.state_size))  # Episode_lengthx64x64x4

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            update_inputs[i, :, :, :] = self.states[i]

        # Prediction of state values for each state appears in the episode
        values = self.critic.predict(update_inputs)

        # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            advantages[i][self.actions[i]] = discounted_rewards[i] - values[i]

        actor_loss = self.actor.fit(update_inputs, advantages, epochs=1, verbose=0)
        critic_loss = self.critic.fit(update_inputs, discounted_rewards, epochs=1, verbose=0)

        self.states, self.actions, self.rewards = [], [], []

        return actor_loss.history['loss'], critic_loss.history['loss']

    def shape_reward(self, reward, misc, prev_misc, t):

        # Check any kill count
        if (misc[0] > prev_misc[0]): #KILLCOUNT
            reward = reward + 1

        if (misc[1] < prev_misc[1]):  # Use ammo
            reward = reward - 0.1

        if (misc[2] < prev_misc[2]):  # Loss HEALTH
            reward = reward - 0.1

        return reward

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5", overwrite=True)
        self.critic.save_weights(name + "_critic.h5", overwrite=True)

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5", overwrite=True)
        self.critic.load_weights(name + "_critic.h5", overwrite=True)

    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
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
        # Start training
        t = 0
        max_life = 0  # Maximum episode life (Proxy for agent performance)

        # Buffer to compute rolling statistics
        life_buffer, ammo_buffer, kills_buffer = [], [], []

        for e in range(self.EPISODES):

            state = self.env.reset()

            # 1x64x64x4
            misc = self.env.get_variables()
            prev_misc = misc

            life = 0  # Episode life

            while not self.env.game.is_episode_finished():
                loss = 0  # Training Loss at each update

                # Sample action from stochastic softmax policy
                action = self.get_action(state)  # (1, 64, 64, 4)

                next_state, reward, done, _ = self.env.step(action, self.frame_per_action)

                if done:
                    # Save max_life
                    if (life > max_life):
                        max_life = life
                    life_buffer.append(life)
                    ammo_buffer.append(misc[1])
                    kills_buffer.append(misc[0])
                    print("Episode Finish ", prev_misc)
                else:
                    life += 1
                    state = next_state
                    misc = self.env.get_variables()

                # Reward Shaping
                reward = self.shape_reward(reward, misc, prev_misc, t)

                # Save trajactory sample <s, a, r> to the memory
                self.append_sample(state, action, reward)

                # Update the cache
                t += 1
                prev_misc = misc

                if (done and t > self.observe):
                    # Every episode, agent learns from sample returns
                    loss = self.learn()

                # Save model every 50 iterations
                if t % 50 == 0:
                    print("Save model")
                    self.save_model("models/a2c")

                state_mode = ""
                if t <= self.observe:
                    state_mode = "Observe mode"
                else:
                    state_mode = "Train mode"

                if (done):
                    average = self.PlotModel(reward, e)

                    # Print performance statistics at every episode end
                    print("TIME", t, "/ GAME", e, "/ STATE", state_mode, "/ ACTION", action, "/ REWARD", reward,
                          "/ LIFE",
                          max_life, "/ LOSS", loss)

                    # Save Agent's Performance Statistics
                    if e % self.stats_window_size == 0 and t > self.observe:
                        print("Update Rolling Statistics")
                        self.mavg_score.append(np.mean(np.array(life_buffer)))
                        self.var_score.append(np.var(np.array(life_buffer)))
                        self.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                        self.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                        # Reset rolling stats buffer
                        life_buffer, ammo_buffer, kills_buffer = [], [], []

                        # Write Rolling Statistics to file

                        with open(self.statistics_file, "a") as stats_file:
                            stats_file.write('Game: ' + str(e) + '\n')
                            stats_file.write('Max Score: ' + str(max_life) + '\n')
                            stats_file.write('mavg_score: ' + str(self.mavg_score) + '\n')
                            stats_file.write('var_score: ' + str(self.var_score) + '\n')
                            stats_file.write('mavg_ammo_left: ' + str(self.mavg_ammo_left) + '\n')
                            stats_file.write('mavg_kill_counts: ' + str(self.mavg_kill_counts) + '\n\n')