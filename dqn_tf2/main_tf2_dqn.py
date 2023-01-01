import numpy as np
import gym
from simple_dqn_tf2 import Agent
from utils.utils import plotLearning
import tensorflow as tf
import vizdoom as vzd
from skimage import transform
from collections import deque

def preprocess_frame(frame):
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


stack_size = 4  # We stack 4 frames
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
def stack_frames(stacked_frames, state, is_new_episode):

    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)
    # print("stacked_state.shape:"+str(stacked_state.shape))
    return stacked_state

class DoomEnv():
    def __init__(self):
        self.game = vzd.DoomGame()
        self.game.load_config("basic.cfg")
        self.game.set_doom_scenario_path("basic.wad")
        self.game.init()

    def create_env(self):
        # Here our possible actions
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        possible_actions = [left, right, shoot]

        return possible_actions

    def step(self, action):

        next_state = stack_frames(stacked_frames, self.game.get_state().screen_buffer, False)
        reward = self.game.make_action(possible_actions[action])
        done = self.game.is_episode_finished()
        info =""
        return next_state, reward, done, info

    def reset(self):
        self.game.new_episode()
        return stack_frames(stacked_frames, self.game.get_state().screen_buffer, False)


if __name__ == '__main__':
    env = DoomEnv()
    possible_actions = env.create_env()
    #hyperparameters
    lr = 0.001
    n_games = 50
    n_actions = 3
    action_shape = (4, 84, 84)
    max_steps = 30

    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr,
                input_dims=action_shape,
                n_actions=n_actions, mem_size=100000, batch_size=64,
                epsilon_end=0.01)
    scores = []
    eps_history = []

    for i in range(n_games):
        done = False
        score = 0
        state = env.reset()
        step = 0
        print('Starting episode: ', i, ', epsilon: %.4f' % agent.epsilon + ' ')
        while step < max_steps:
            step += 1
            if not done:
                action = agent.choose_action(state)
                nex_state, reward, done, info = env.step(action)
                score += reward
                agent.store_transition(state, action, reward, nex_state, done)
                state = nex_state
                agent.learn()

        print('Steps taken:', step)
        step = max_steps
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('Episode finished:', i, 'score %.2f' % score,
                'average_score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    env.game.close()
    filename = '50_vizdoom_tf2.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)