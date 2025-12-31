from doom_agent.algorithms.a3c.base.agent import A3CAgent
from doom_agent.paths import scenario_path
from doom_agent.wrappers.doom_wrapper import DoomEnv
from doom_agent.utils.utils import set_gpu_memory_size

if __name__ == "__main__":

    set_gpu_memory_size(4096)

    env_name = 'Vizdoom-v0'
    state_size = (64, 64, 4)
    env = DoomEnv(stack_size=4, img_shape=(64, 64), scenario=scenario_path("deadly_corridor.cfg"))
    env.create_env()
    agent = A3CAgent(env_name, env, state_size)
    agent.train(n_threads=2)
