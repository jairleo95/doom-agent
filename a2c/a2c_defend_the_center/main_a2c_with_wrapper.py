#https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py
#!/usr/bin/env python
from __future__ import print_function

from datetime import datetime

import skimage as skimage
from skimage import transform, color
import numpy as np

from vizdoom import *
import tensorflow as tf

from a2c.a2c_defend_the_center.agent import A2CAgent
from wrappers.doom_wrapper import DoomEnv


def set_gpu_memory_size():
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

if __name__ == "__main__":
    set_gpu_memory_size()

    env = DoomEnv(stack_size=4,
                  img_shape=(64, 64),
                  crop_args=[15, -5, 20, -20],
                  scenario="defend_the_center.cfg",
                  resolution=ScreenResolution.RES_640X480, img_channel="last"
                  )

    #statistics
    statistics_file = "statistics/a2c_stats_run_" + datetime.now().strftime("%y-%m-%d-%H-%M") + ".txt"

    action_size, action_shape = env.create_env()

    game = env.game

    game.new_episode()
    misc = env.get_variables()  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    img_rows, img_cols = 64, 64
    img_channels = 4  # We stack 4 frames

    state_size = (img_rows, img_cols, img_channels)
    agent = A2CAgent(env, state_size, action_size, statistics_file)
    agent.train()
