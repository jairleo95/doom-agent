#https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py
#!/usr/bin/env python
from __future__ import print_function

from datetime import datetime

from vizdoom import *

from a2c_defend_the_center.agent import A2CAgent
from wrappers.doom_wrapper import DoomEnv
from utils.utils import set_gpu_memory_size


if __name__ == "__main__":
    set_gpu_memory_size(memory_limit=2084)

    env = DoomEnv(stack_size=4,
                  img_shape=(64, 64),
                  crop_args=[15, -5, 20, -20],
                  scenario="../scenarios/defend_the_center.cfg",
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
