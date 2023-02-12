#https://github.com/jairleo95/Reinforcement_Learning_by_pythonlessons/blob/c9717f523fb9bd4bb8ccb5b34bd6ee6c76ea21b6/05_CartPole-reinforcement-learning_PER_D3QN/Cartpole_PER_D3QN_TF2.py#L195
#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
#(a3c vizdoom)
#https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
#https://github.com/GoingMyWay/ViZDoomAgents

from datetime import datetime

# tensorboardX
from tensorboardX import SummaryWriter

from wrappers.doom_wrapper import *
from dddqn_tf2_v2.agent import Agent
from dddqn_tf2_v2.config import *
from utils.utils import *
if __name__ == '__main__':
    set_gpu_memory_size(memory_limit=2048)

    # Summary writer de TensorBoardX
    summary_filename = "logs/dddqn_tf2_vizdoom" + datetime.now().strftime("%y-%m-%d-%H-%M")
    writer = SummaryWriter(summary_filename)

    # Create an environment of doom
    env = DoomEnv(stack_size=stack_size,
                  img_shape=img_shape,
                  crop_args=[15, -5, 20, -20],
                  scenario="../scenarios/deadly_corridor.cfg",
                  color_mode="GRAY",
                  img_channel="last")
    num_actions, action_shape = env.create_env()

    agent = Agent(env=env,
                  gamma=gamma,
                  lr=learning_rate,
                  epsilon=explore_start,
                  epsilon_end=explore_stop,
                  epsilon_dec=decay_rate,
                  n_actions=num_actions,
                  state_size=state_size,
                  action_shape=action_shape,
                  mem_size=memory_size,
                  batch_size=batch_size,
                  total_episodes=total_episodes,
                  max_steps=max_steps,
                  pretrain_length = pretrain_length,
                  writer=writer)

if training:
    agent.train()

agent.test()