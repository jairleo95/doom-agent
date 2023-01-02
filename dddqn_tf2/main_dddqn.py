import os

import sys
import random
import numpy as np
import vizdoom as vzd
from datetime import datetime
from skimage import transform
from collections import deque
from utils.utils import plotLearning
# tensorboardX
from tensorboardX import SummaryWriter

from dddqn_tf2.agent import Agent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Summary writer de TensorBoardX
summary_filename = "logs/dddqn_tf2_vizdoom" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_filename)

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

    ### MODEL HYPERPARAMETERS
    # state_size = [100, 120, 4]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
    action_size = env.game.get_available_buttons_size()  # 7 possible actions
    learning_rate = 0.00025  # Alpha (aka learning rate)
    n_actions = 3
    state_size = (4, 84, 84)
    action_shape = [0,1,2]

    ### TRAINING HYPERPARAMETERS
    total_episodes = 100  # number of games
    max_steps = 30
    batch_size = 64

    # FIXED Q TARGETS HYPERPARAMETERS
    max_tau = 10000  # Tau is the C step where we update our target network

    # EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
    explore_start = 1.0  # exploration probability at start
    explore_stop = 0.01  # minimum exploration probability
    decay_rate = 0.00005  # exponential decay rate for exploration prob

    # Q LEARNING hyperparameters
    gamma = 0.95  # Discounting rate

    ### MEMORY HYPERPARAMETERS
    ## If you have GPU change to 1million
    pretrain_length = 100000  # Number of experiences stored in the Memory when initialized for the first time
    memory_size = 100000  # Number of experiences the Memory can keep

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
    episode_render = False

    possible_actions = env.create_env()

    agent = Agent(gamma=gamma,lr=learning_rate,
                  epsilon=explore_start,epsilon_end=explore_stop,epsilon_dec=decay_rate,
                  n_actions=n_actions, input_dims=state_size,mem_size=memory_size,
                  batch_size=batch_size, replace=max_tau)
    scores = []
    eps_history = []

    print("Filling memory: ", pretrain_length)
    for i in range(pretrain_length):
        sys.stdout.write(f"\r{str((i/pretrain_length)*100)} %")
        if i == 0:
            # First we need a state
            state = env.reset()

        # Random action
        action = random.choice(action_shape)
        # Get the rewards-
        nex_state, reward, done, _ = env.step(action)

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            agent.store_transition(state, action, reward, next_state, done)
            # Start a new episode
            state = env.reset()

        else:
            next_state = env.game.get_state().screen_buffer
            next_state = stack_frames(stacked_frames, next_state, False)
            agent.store_transition(state, action, reward, next_state, done)
            # Our state is now the next_state
            state = next_state
        i += 1
    print('\nDone initializing memory')

if training:

    episode_rewards = list()
    for episode in range(total_episodes):
        total_reward = 0.0
        done = False
        score = 0
        step = 0

        print('Starting episode: ', episode, ', epsilon: %.4f' % agent.epsilon + ' ')
        # Start a new episode
        state = env.reset()
        print("[Steps]")
        steps_taken = 0
        while step < max_steps:
            step += 1
            if not done:
                # env.render()
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                steps_taken = step

                episode_rewards.append(reward)
                sys.stdout.write(f"\r{step}")
                
                agent.store_transition(state, action, reward, next_state, done)
                state = next_state
                agent.train()

            # total_reward += reward
        print("\n[DONE]")

        print('Steps taken:', step)
        step = max_steps
        episode_rewards.append(score)
        eps_history.append(agent.epsilon)
        scores.append(score)

    env.game.close()
    avg_score = np.mean(scores[-100:])
    print('Episode finished:', episode, 'score %.2f' % score,
          'iterations: {}'.format(steps_taken),
          'average_score %.2f' % avg_score,
          'epsilon %.2f' % agent.epsilon)

    writer.add_scalar("main/ep_reward", score, episode)
    # writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), episode)
    # writer.add_scalar("main/max_ep_reward", agent.best_reward, episode)

    # Save model every 5 episodes
    if episode % 5 == 0:
        agent.save_model()
        print("Models saved")

    filename = str(total_episodes) + 'Games' + 'Gamma' + str(agent.gamma) + 'Alpha' + str(agent.lr) + 'Memory' + str(
        agent.mem_size) + '_vizdoom_dddqn__tf2.png'
    x = [i + 1 for i in range(total_episodes)]
    plotLearning(x, scores, eps_history, filename)
