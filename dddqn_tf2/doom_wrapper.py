import numpy as np
import vizdoom as vzd
from skimage import transform
from collections import deque

class DoomEnv():
    def __init__(self, stack_size, img_shape):
        self.game = vzd.DoomGame()
        self.stack_size = stack_size
        self.img_shape = img_shape
        self.possible_actions = None
        self.stacked_frames = None

    def create_env(self):
        # Here our possible actions
        self.game.load_config("basic.cfg")
        self.game.set_doom_scenario_path("basic.wad")
        self.game.init()

        num_actions = self.game.get_available_buttons_size()
        action_shape = list(range(0, num_actions - 1))
        # Here we create an hot encoded version of our actions (x possible actions)
        # possible_actions = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]...]
        self.possible_actions = np.identity(num_actions, dtype=int).tolist()
        self.stacked_frames = deque([np.zeros(self.img_shape, dtype=int) for i in range(self.stack_size)], maxlen=4)

        return num_actions, action_shape

    def step(self, action):

        # next_state = self.stack_frames(self.stacked_frames, self.game.get_state().screen_buffer, False)
        reward = self.game.make_action(self.possible_actions[action])
        done = self.game.is_episode_finished()
        # info =""
        # return next_state, reward, done, info
        return reward, done

    def reset(self):
        self.game.new_episode()
        # return self.stack_frames(self.stacked_frames, self.game.get_state().screen_buffer, False)

    def preprocess_frame(self, frame):
        # Crop the screen (remove the roof because it contains no information)
        cropped_frame = frame[30:-10, 30:-30]
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
            stacked_frames = deque([np.zeros(self.img_shape, dtype=int) for i in range(self.stack_size)], maxlen=4)

            # Because we're in a new episode, copy the same frame 4x
            for i in range(self.stack_size):
                stacked_frames.append(frame)
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)
            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2)
        # print("stacked_state.shape:"+str(stacked_state.shape))
        return stacked_state
