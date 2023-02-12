#https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py
#!/usr/bin/env python
from __future__ import print_function

from datetime import datetime

import skimage as skimage
from skimage import transform, color
import numpy as np

from vizdoom import *
import tensorflow as tf

from a2c_defend_the_center.networks import Networks
from a2c_defend_the_center.agent import A2CAgent

def preprocessImg(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img

if __name__ == "__main__":

    # Avoid Tensorflow eats up GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    #statistics
    statistics_file = "statistics/a2c_stats_run_" + datetime.now().strftime("%y-%m-%d-%H-%M") + ".txt"

    game = DoomGame()
    game.load_config("../../scenarios/defend_the_center.cfg")
    game.set_sound_enabled(False)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.init()

    # Maximum number of episodes
    max_episodes = 300

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows, img_cols = 64, 64
    # Convert image into Black and white
    img_channels = 4  # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = A2CAgent(state_size, action_size)
    agent.actor = Networks.actor_network(state_size, action_size, agent.actor_lr)
    agent.critic = Networks.critic_network(state_size, agent.value_size, agent.critic_lr)

    # Start training
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    for i in range(max_episodes):

        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables
        prev_misc = misc

        x_t = game_state.screen_buffer  # 480 x 640
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(([x_t] * 4), axis=2)  # It becomes 64x64x4
        s_t = np.expand_dims(s_t, axis=0)  # 1x64x64x4

        life = 0  # Episode life

        while not game.is_episode_finished():

            loss = 0  # Training Loss at each update
            r_t = 0  # Initialize reward at time t
            a_t = np.zeros([action_size])  # Initialize action at time t

            x_t = game_state.screen_buffer
            x_t = preprocessImg(x_t, size=(img_rows, img_cols))
            x_t = np.reshape(x_t, (1, img_rows, img_cols, 1))
            s_t = np.append(x_t, s_t[:, :, :, :3], axis=3)

            print("train.s_t.shape:", s_t.shape)#(1, 64, 64, 4)

            # Sample action from stochastic softmax policy
            action_idx, policy = agent.get_action(s_t)
            a_t[action_idx] = 1

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action  # Frame Skipping = 4
            game.advance_action(skiprate)

            r_t = game.get_last_reward()  # Each frame we get reward of 0.1, so 4 frames will be 0.4
            # Check if episode is terminated
            is_terminated = game.is_episode_finished()

            if (is_terminated):
                # Save max_life
                if (life > max_life):
                    max_life = life
                life_buffer.append(life)
                ammo_buffer.append(misc[1])
                kills_buffer.append(misc[0])
                print("Episode Finish ", prev_misc, policy)
            else:
                life += 1
                game_state = game.get_state()  # Observe again after we take the action
                misc = game_state.game_variables

            # Reward Shaping
            r_t = agent.shape_reward(r_t, misc, prev_misc, t)

            # Save trajactory sample <s, a, r> to the memory
            agent.append_sample(s_t, action_idx, r_t)

            # Update the cache
            t += 1
            prev_misc = misc

            if (is_terminated and t > agent.observe):
                # Every episode, agent learns from sample returns
                loss = agent.learn()

            # Save model every 10000 iterations
            if t % 10000 == 0:
                print("Save model")
                agent.save_model("models/a2c")

            state = ""
            if t <= agent.observe:
                state = "Observe mode"
            else:
                state = "Train mode"

            if (is_terminated):

                # Print performance statistics at every episode end
                print("TIME", t, "/ GAME", GAME, "/ STATE", state, "/ ACTION", action_idx, "/ REWARD", r_t, "/ LIFE",
                      max_life, "/ LOSS", loss)

                # Save Agent's Performance Statistics
                if GAME % agent.stats_window_size == 0 and t > agent.observe:
                    print("Update Rolling Statistics")
                    agent.mavg_score.append(np.mean(np.array(life_buffer)))
                    agent.var_score.append(np.var(np.array(life_buffer)))
                    agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                    agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                    # Reset rolling stats buffer
                    life_buffer, ammo_buffer, kills_buffer = [], [], []

                    # Write Rolling Statistics to file

                    with open(statistics_file, "a") as stats_file:
                        stats_file.write('Game: ' + str(GAME) + '\n')
                        stats_file.write('Max Score: ' + str(max_life) + '\n')
                        stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                        stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                        stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                        stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

        # Episode Finish. Increment game count
        GAME += 1
