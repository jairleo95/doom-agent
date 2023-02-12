#https://github.com/nicknochnack/DoomReinforcementLearning/blob/main/VizDoom-DeadlyCorridor-Tutorial.ipynb

# Import environment base class from OpenAI Gym
import random
from vizdoom import *
from gym import Env
import numpy as np
from gym.spaces import Discrete, Box
import cv2

# Create Vizdoom OpenAI Gym Environment
class VizDoomGym(Env):
    # Function that is called when we start the env
    def __init__(self, state_shape, config):
        # Inherit from Env
        super().__init__()
        # Setup the game
        self.game = DoomGame()
        self.game.load_config(config)

        self.state_shape = state_shape

        # Start the game
        self.game.init()

        # Create the action space and observation space
        self.num_actions = self.game.get_available_buttons_size()
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Box(low=0, high=255, shape=state_shape, dtype=np.uint8)

        # Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52  ## CHANGED

    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step
        actions = np.identity(self.num_actions)
        reward = self.game.make_action(actions[action], 4)# 4 tics timeout
        info = 0

        # Get all the other stuff we need to retun
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)

            # Reward shaping
            # game_variables = self.game.get_state().game_variables
            # health, damage_taken, hitcount, ammo = game_variables
            #
            # # Calculate reward deltas
            # damage_taken_delta = -damage_taken + self.damage_taken
            # self.damage_taken = damage_taken
            # hitcount_delta = hitcount - self.hitcount
            # self.hitcount = hitcount
            # ammo_delta = ammo - self.ammo
            # self.ammo = ammo
            #
            # reward = reward + damage_taken_delta * 10 + hitcount_delta * 200 + ammo_delta * 5
            # info = ammo
        else:
            state = np.zeros(self.observation_space.shape)

        info = {"info": info}
        done = self.game.is_episode_finished()

        return state, reward, done, info

        # Define how to render the game or environment

    def render(self):
        pass

    # What happens when we start a new game
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    # Grayscale the game frame and resize it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (self.state_shape[0], self.state_shape[1]), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, self.state_shape)
        return state

    # Call to close down the game
    def close(self):
        self.game.close()


if __name__ == '__main__':
    env = VizDoomGym(state_shape=(100, 160, 1),
                     config="../scenarios/deadly_corridor.cfg")
    env.game.new_episode()
    action_shape = list(range(0, env.num_actions))

    for i in range(10):
        state = env.reset()
        done = False
        # frame = env.get_image(next_state, False)
        # env.imshow(next_state, 0)
        # time.sleep(3)

        while not done:
            action = random.choice(action_shape)
            next_state, reward, done, _ = env.step(action)
            print("reward: ", reward)
            if done:
                # print("TOTAL Reward: ", total_reward)
                break