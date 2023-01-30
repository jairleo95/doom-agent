#https://pylessons.com/A2C-reinforcement-learning
#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
#https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py

#Advantage Actor-Critic implementation
from a2c.agent import A2CAgent
from wrappers.doom_wrapper import DoomEnv

if __name__ == "__main__":
    env_name = 'Vizdoom-v0'
    state_size = (4, 64, 64)
    env = DoomEnv(stack_size=4, img_shape=(64, 64))
    num_actions, action_shape = env.create_env()
    agent = A2CAgent(env_name, env, state_size, num_actions)
    agent.train()
    # agent.test('Models/Vizdoom-v0_A2C_2.5e-05_Actor.h5', '')