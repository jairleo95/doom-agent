from a3c.agent import A3CAgent
from wrappers.doom_wrapper import DoomEnv
from utils.utils import set_gpu_memory_size

if __name__ == "__main__":

    set_gpu_memory_size(4096)

    env_name = 'Vizdoom-v0'
    state_size = (64, 64, 4)
    env = DoomEnv(stack_size=4, img_shape=(64, 64), scenario="../scenarios/deadly_corridor.cfg")
    env.create_env()
    agent = A3CAgent(env_name, env, state_size)
    agent.train(n_threads=2)