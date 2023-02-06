import os

import sys
import random
import numpy as np
from datetime import datetime
from wrappers.doom_wrapper import DoomEnv

from vizdoom.vizdoom import DoomGame

from utils.utils import plotLearning
# tensorboardX
from tensorboardX import SummaryWriter

from ddpg_tf2.agent import Agent
from ddpg_tf2.config import *

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
                  batch_size=batch_size, total_episodes=total_episodes, writer= writer)


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
    agent.train()


agent.load_model("Models")
game = DoomGame()

# Load the correct configuration (TESTING)
game.load_config("basic.cfg")
# Load the correct scenario (in our case deadly_corridor scenario)
game.set_doom_scenario_path("basic.wad")
game.init()


for i in range(100):

    game.new_episode()
    state = game.get_state().screen_buffer
    state = env.stack_frames(env.stacked_frames, state, True)
    state = np.reshape(state, [1, *agent.state_size])

    while not game.is_episode_finished():
        # Estimate the Qs values state
        # Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        Qs = agent.model.predict(state)

        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = env.possible_actions[int(choice)]

        game.make_action(action)
        done = game.is_episode_finished()

        if done:
            break

        else:
            next_state = game.get_state().screen_buffer
            next_state = env.stack_frames(env.stacked_frames, next_state, False)
            next_state = np.reshape(next_state, [1, *agent.state_size])
            state = next_state

    score = game.get_total_reward()
    print("Score: ", score)

game.close()