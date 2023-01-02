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
    env = DoomEnv(stack_size=4, img_shape=(84, 84))
    num_actions, action_shape = env.create_env()

    agent = Agent(gamma=gamma, lr=learning_rate,
                  epsilon=explore_start, epsilon_end=explore_stop, epsilon_dec=decay_rate,
                  n_actions=num_actions, input_dims=state_size, mem_size=memory_size,
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
        # Get the rewards
        nex_state, reward, done, _ = env.step(action)

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            agent.add_experience(state, action, reward, next_state, done)
            # Start a new episode
            state = env.reset()

        else:
            next_state = env.game.get_state().screen_buffer
            next_state = env.stack_frames(env.stacked_frames, next_state, False)
            agent.add_experience(state, action, reward, next_state, done)
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
                
                agent.add_experience(state, action, reward, next_state, done)
                state = next_state
                agent.learn()

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

    # writer.add_scalar("main/ep_reward", score, episode)
    # writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), episode)
    # writer.add_scalar("main/max_ep_reward", agent.best_reward, episode)

    # Save model every 5 episodes
    if episode % 5 == 0:
        agent.save_model("model")
        print("Models saved")

    filename = 'results/'+str(total_episodes) + 'Games' + 'Gamma' + str(agent.gamma) + 'Alpha' + str(learning_rate) + 'Memory' + str(
        memory_size) + '_vizdoom_dddqn__tf2.png'
    x = [i + 1 for i in range(total_episodes)]
    plotLearning(x, scores, eps_history, filename)
