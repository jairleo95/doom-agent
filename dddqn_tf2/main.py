#https://github.com/jairleo95/Reinforcement_Learning_by_pythonlessons/blob/c9717f523fb9bd4bb8ccb5b34bd6ee6c76ea21b6/05_CartPole-reinforcement-learning_PER_D3QN/Cartpole_PER_D3QN_TF2.py#L195
#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb

import os

import sys
import random
import numpy as np
from datetime import datetime
from utils.utils import plotLearning
# tensorboardX
from tensorboardX import SummaryWriter

from dddqn_tf2.doom_wrapper import *
from dddqn_tf2.dddqn_agent import Agent
from dddqn_tf2.config import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Summary writer de TensorBoardX
summary_filename = "logs/dddqn_tf2_vizdoom" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_filename)


if __name__ == '__main__':

    # Create an environment of doom
    REM_STEP = 4
    env = DoomEnv(stack_size=4, img_shape=(84, 84))
    num_actions, action_shape = env.create_env()

    agent = Agent(gamma=gamma, lr=learning_rate,
                  epsilon=explore_start, epsilon_end=explore_stop, epsilon_dec=decay_rate,
                  n_actions=num_actions, state_size=state_size, mem_size=memory_size,
                  batch_size=batch_size, replace=max_tau)
    scores = []
    eps_history = []

    print("Filling memory: ", pretrain_length)
    for i in range(pretrain_length):
        sys.stdout.write(f"\r{str((i/pretrain_length)*100)} %")
        if i == 0:
            # First we need a state
            state = env.stack_frames(env.stacked_frames, env.game.get_state().screen_buffer, True)

        # Random action
        action = random.choice(action_shape)
        # Get the rewards
        reward, done = env.step(action)

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            agent.remember(state, action, reward, next_state, done)
            # Start a new episode
            env.game.new_episode()

            # First we need a state
            state = env.game.get_state().screen_buffer

            # Stack the frames
            state = env.stack_frames(env.stacked_frames, state, True)

        else:
            next_state = env.game.get_state().screen_buffer
            next_state = env.stack_frames(env.stacked_frames, next_state, False)
            agent.remember(state, action, reward, next_state, done)
            # Our state is now the next_state
            state = next_state
    print('\nDone initializing memory')

if training:

    episode_rewards = list()
    decay_step = 0

    # Init the game
    env.game.init()

    # Update the parameters of our TargetNetwork with DQN_weights

    for episode in range(total_episodes):
        done = False
        step = 0

        # Initialize the rewards of the episode
        episode_rewards = []

        print('Starting episode: ', episode, ', epsilon: %.4f' % agent.epsilon + ' ')
        # Start a new episode
        env.game.new_episode()
        state = env.game.get_state().screen_buffer

        # Remember that stack frame function also call our preprocess function.
        state = env.stack_frames(env.stacked_frames, state, True)

        while step < max_steps:
            step += 1
            decay_step += 1

            print('state.shae', state.shape)
            action, explore_probability = agent.act(state, decay_step)

            # Do the action
            reward, done = env.step(action)

            # Add the reward to total reward
            episode_rewards.append(reward)

            if done:
                # the episode ends so no next state
                next_state = env.stack_frames(env.stacked_frames, np.zeros(env.img_shape, dtype=np.int), False)
                # every step update target model
                agent.update_target_model()

                step = max_steps

                # Get the total reward of the episode
                total_reward = np.sum(episode_rewards)

                eps_history.append(agent.epsilon)
                scores.append(total_reward)
                # every episode, plot the result
                average = agent.PlotModel(total_reward, episode)
                print('Episode: {}/{}, Total reward: {}, e: {:.2}, average: {}'.format(episode, total_episodes, total_reward,
                                                                                explore_probability, average))
                agent.remember(state, action, reward, next_state, done)
                # if i == env._max_episode_steps:
                #     print("Saving trained model to", "DQN")
                #     # self.save(self.Model_name)
                #     break
            else:
                next_state = env.stack_frames(env.stacked_frames, env.game.get_state().screen_buffer, False)

                agent.remember(state, action, reward, next_state, done)

                state = next_state
            agent.learn()
    env.game.close()

    filename = 'results/'+str(total_episodes) + 'Games' + 'Gamma' + str(agent.gamma) + 'Alpha' + str(learning_rate) + 'Memory' + str(
        memory_size) + '_vizdoom_dddqn__tf2.png'
    x = [i + 1 for i in range(total_episodes)]
    plotLearning(x, scores, eps_history, filename)
