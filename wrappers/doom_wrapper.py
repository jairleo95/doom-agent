import random

import time
import cv2
import numpy as np
import vizdoom as vzd
from skimage import transform
from collections import deque

from vizdoom.vizdoom import ScreenResolution


class DoomEnv():
    def __init__(self, stack_size, img_shape,
                 crop_args=[30, -10, 30, -30],
                 scenario ="basic.cfg",
                 resolution=ScreenResolution.RES_160X120, color_mode="GRAY", img_channel="last"):
        self.game = vzd.DoomGame()
        self.scenario = scenario
        self.resolution = resolution
        self.stack_size = stack_size
        self.img_shape = img_shape
        self.possible_actions = None
        self.stacked_frames = None
        self.is_new_episode = True
        self.crop_args = crop_args
        self.color_mode = color_mode
        self.img_channel = img_channel

    def create_env(self):
        # Here our possible actions
        self.game.load_config("../../scenarios/"+self.scenario)
        # self.game.set_sound_enabled(True)
        # self.game.set_doom_scenario_path("../scenarios/defend_the_center.wad")
        self.game.set_screen_resolution(self.resolution)
        # set color mode
        if self.color_mode == 'RGB':
            self.game.set_screen_format(vzd.ScreenFormat.CRCGCB)
            # self.num_channels = 3
        elif self.color_mode == 'GRAY':
            self.game.set_screen_format(vzd.ScreenFormat.GRAY8)
            # self.num_channels = 1
        else:
            print("Unknown color mode")
            raise

        self.game.init()

        num_actions = self.game.get_available_buttons_size()
        action_shape = list(range(0, num_actions - 1))
        # Here we create an hot encoded version of our actions (x possible actions)
        # possible_actions = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]...]
        self.possible_actions = np.identity(num_actions, dtype=int).tolist()
        self.stacked_frames = deque([np.zeros(self.img_shape, dtype=np.int) for i in range(self.stack_size)], maxlen=4)

        return num_actions, action_shape

    def step(self, action):

        next_state = self.get_image(self.get_state())
        reward = self.game.make_action(self.possible_actions[action])
        done = self.game.is_episode_finished()
        info =""
        return next_state, reward, done, info

    def step(self, action, frame_per_action):
        next_state = self.get_image(self.get_state())
        self.game.set_action(self.possible_actions[action])
        self.game.advance_action(frame_per_action)
        reward = self.game.get_last_reward()
        done = self.game.is_episode_finished()
        info = ""
        return next_state, reward, done, info

    def get_state(self):
        return self.game.get_state().screen_buffer

    def get_variables(self):
        return self.game.get_state().game_variables

    def reset(self):
        self.game.new_episode()
        state = self.get_image(self.get_state(), True)
        return state

    def preprocess_frame(self, frame):
        # Crop the screen (remove the roof because it contains no information)
        args = self.crop_args
        cropped_frame = frame[args[0]:args[1], args[2]:args[3]]
        # Normalize Pixel Values
        normalized_frame = cropped_frame / 255.0
        # Resize
        preprocessed_frame = transform.resize(normalized_frame, self.img_shape)
        return preprocessed_frame

    def stack_frames(self, stacked_frames, state, is_new_episode):
        # Preprocess frame
        frame = self.preprocess_frame(state)

        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros(self.img_shape, dtype=np.int) for i in range(self.stack_size)], maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            for i in range(self.stack_size):
                stacked_frames.append(frame)
            # Stack the frames
            if self.img_channel == "last":
                stacked_state = np.stack(stacked_frames, axis=2)
            else:
                stacked_state = np.stack(stacked_frames, axis=0)

            print("stacked_state.shape:", stacked_state.shape)
            # self.imshow(np.moveaxis(stacked_state, 2, 0))

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)
            # Build the stacked state (first dimension specifies different frames)
            if self.img_channel == "last":
                stacked_state = np.stack(stacked_frames, axis=2)
            else:
                stacked_state = np.stack(stacked_frames, axis=0)

            print("stacked_state.shape:", stacked_state.shape)
        return stacked_state

    def get_image(self, frame, is_new_episode=False):
        #stack frames
        image = self.stack_frames(self.stacked_frames, frame, is_new_episode)
        return image

    def imshow(self, image, rem_step=0):
        if self.img_channel == "last":
            image = np.moveaxis(image, 2, 0)
        cv2.imshow("Image", image[rem_step, ...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return


if __name__ == '__main__':
    env = DoomEnv(stack_size=4,
                  img_shape=(64, 64),
                  crop_args=[15, -5, 20, -20],
                  scenario="defend_the_center.cfg",
                  resolution=ScreenResolution.RES_640X480, img_channel="last"
                  )
    num_actions, action_shape = env.create_env()
    env.game.new_episode()

    for i in range(1000):
        state = env.get_state()

        action = random.choice(action_shape)
        next_state, reward, done, _ = env.step(action)

        # frame = env.get_image(next_state, False)

        env.imshow(next_state, 0)
        time.sleep(3)

        if done:
            state = env.reset()
