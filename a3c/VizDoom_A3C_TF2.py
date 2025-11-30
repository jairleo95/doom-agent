import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import threading
from threading import Lock
import time
from wrappers.doom_wrapper import DoomEnv
from vizdoom.vizdoom import ScreenResolution

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)

    # CNN Layers for image processing
    X = Conv2D(32, 8, strides=(4, 4), padding="valid", activation="elu", data_format="channels_last")(X_input)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="elu", data_format="channels_last")(X)
    X = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="elu", data_format="channels_last")(X)
    X = Flatten()(X)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs=X_input, outputs=action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs=X_input, outputs=value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic

class A3CAgent:
    def __init__(self, scenario='defend_the_center.cfg'):
        self.scenario = scenario
        # Create a dummy env to get action size and shape
        temp_env = DoomEnv(stack_size=4, img_shape=(80, 80), scenario=scenario, resolution=ScreenResolution.RES_160X120, img_channel="last")
        self.action_size, _ = temp_env.create_env()
        temp_env.game.close()

        self.EPISODES, self.episode, self.max_average = 20000, 0, -21.0 
        self.lock = Lock()
        self.lr = 0.000025

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4
        self.state_size = (self.ROWS, self.COLS, self.REM_STEP) # Channels last

        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models_VizDoom'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(self.scenario.replace('.cfg', ''), self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)

    def act(self, state):
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        return action

    def discount_rewards(self, reward):
        gamma = 0.99
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def replay(self, states, actions, rewards):
        states = np.vstack(states)
        actions = np.vstack(actions)

        discounted_r = self.discount_rewards(rewards)
        value = self.Critic.predict(states)[:, 0]
        advantages = discounted_r - value

        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        # self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        # self.Critic.save(self.Model_name + '_Critic.h5')

    def train(self, n_threads):
        envs = [DoomEnv(stack_size=4, img_shape=(80, 80), scenario=self.scenario, resolution=ScreenResolution.RES_160X120, img_channel="last") for i in range(n_threads)]
        for env in envs:
            env.create_env()

        threads = [threading.Thread(
            target=self.train_threading,
            daemon=True,
            args=(self,
                  envs[i],
                  i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()

        for t in threads:
            time.sleep(10)
            t.join()

    def train_threading(self, agent, env, thread):
        while self.episode < self.EPISODES:
            score, done, SAVING = 0, False, ''
            state = env.reset()
            state = np.expand_dims(state, axis=0) # Add batch dimension
            states, actions, rewards = [], [], []
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.expand_dims(next_state, axis=0)

                states.append(state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)

                score += reward
                state = next_state

            self.lock.acquire()
            self.replay(states, actions, rewards)
            self.lock.release()

            with self.lock:
                self.scores.append(score)
                self.episodes.append(self.episode)
                self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
                average = self.average[-1]

                if average >= self.max_average:
                    self.max_average = average
                    self.save()
                    SAVING = "SAVING"
                else:
                    SAVING = ""
                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES,
                                                                                         thread, score, average,
                                                                                         SAVING))
                if (self.episode < self.EPISODES):
                    self.episode += 1
        env.game.close()

    def test(self, Actor_name):
        self.load(Actor_name, '')
        env = DoomEnv(stack_size=4, img_shape=(80, 80), scenario=self.scenario, resolution=ScreenResolution.RES_160X120, img_channel="last")
        env.create_env()
        
        for e in range(10):
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            score = 0
            while not done:
                # env.game.set_window_visible(True) # Uncomment to see the game
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = env.step(action)
                state = np.expand_dims(state, axis=0)
                score += reward
                time.sleep(0.02)
            print("episode: {}, score: {}".format(e, score))
        env.game.close()

if __name__ == "__main__":
    agent = A3CAgent(scenario='defend_the_center.cfg')
    # agent.train(n_threads=4)
    # agent.test('Models_VizDoom/defend_the_center_A3C_2.5e-05_Actor.h5')
