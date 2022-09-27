#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning
#articles
#https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#key-concepts
#https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
from builtins import print
import torch
from datetime import datetime
from agent import Agent
import numpy as np
from argparse import ArgumentParser
from collections import deque

#doom
import vizdoom as vzd
import random
from skimage import transform

from utils.params_manager import ParamsManager
from utils.utils import plotLearning

# tensorboardX
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings('ignore')
import sys
import matplotlib.pyplot as plt

# contador global de ejecuciones
global_step_num = 0

def show_image(frame):
    #img = mpimg.imread(frame)
    imgplot = plt.imshow(frame)
    plt.show()


def create_environment():
    game = vzd.DoomGame()
    # Load the correct configuration
    game.load_config("basic.cfg")
    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")
    game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions

def env_step(action):
    reward = game.make_action(action)
    done = game.is_episode_finished()
    return reward, done

def preprocess_frame(frame):
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame


stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)
        #https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/
        #change channel ordering
        #stacked_state = np.moveaxis(stacked_state, 2, 0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)
        #stacked_state = np.moveaxis(stacked_state, 2, 0)
        #show_image(stacked_state)
    #print("stacked_state.shape:"+str(stacked_state.shape))
    return stacked_state, stacked_frames


##Parseador de Argumentos
args = ArgumentParser("DeepQLearning")

# para leer los parametros del json
args.add_argument("--params-file", help="Path del fichero JSON de parametros. El valor por defecto es parameters.json", default="parameters.json", metavar="PFILE")
args.add_argument("--env", help="Entorno de ID de Atari disponible en OpenAI Gym. El valor por defecto sera SeaquestNoFrameskip-v4", default="Pong-v0", metavar="ENV")
args.add_argument("--gpu-id", help="ID de la GPU a utilizar, por defecto 0", default=0, type=int, metavar="GPU_ID")
args.add_argument("--test", help="Modo de testing para jugar sin aprender. Por defecto esta desactivado", action="store_true", default=False)
args.add_argument("--render", help="Renderiza el entorno e pantalla. Desactivado por defecto", action="store_true", default=False)
args.add_argument("--record", help="Almacena vodeos y estados de la performance del agente", action="store_true", default=True)
args.add_argument("--output_dir", help="Directorio para almacenar los outputs. Por defecto = ./trained_models/results")
args = args.parse_args()

# Parametros globales
manager = ParamsManager(args.params_file)
# ficheros de logs acerca de la configuracion de las ejecuciones
summary_filename_prefix = manager.get_agent_params()['summary_filename_prefix']
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

# Summary writer de TensorBoardX
writer = SummaryWriter(summary_filename)
manager.export_agent_params(summary_filename + "/" + "agent_params.json")
manager.export_environment_params(summary_filename + "/" + "environment_params.json")

# contador global de ejecuciones
#global_step_num = 0

print("Cuda is available:"+ str(torch.cuda.is_available()))
use_cuda = manager.get_agent_params()['use_cuda']
#device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")


# habilitar la semilla aleatorio para poder reproducir el experimiento a posteriori
seed = manager.get_agent_params()['seed']
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':

    game, possible_actions = create_environment()

    params = manager.get_agent_params()
    params["test"] = args.test

    agent = Agent(params, maxMemorySize=params["experience_memory_size"], writer= writer)

    # Render the environment
    game.new_episode()
    i = 0

    #Hyperparameters
    ### MODEL HYPERPARAMETERS

    ##Training hyperparameters
    total_episodes = params["total_episodes"]
    max_steps = params["max_steps"]
    batch_size = params['batch_size']

    # Exploration parameters for epsilon greedy strategy

    # Q learning hyperparameters

    ### MEMORY HYPERPARAMETERS

    ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
    training = True

    ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
    episode_render = False


    print("Filling memory: ")
#    for i in range(pretrain_length):
    while agent.memCntr < agent.memSize:
        sys.stdout.write(f"\r{str(i)}")
        if i == 0:
            # First we need a state
            state = game.get_state().screen_buffer #state.shape (120, 160)
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            #show_image(state)

        # Random action
        action = random.choice(possible_actions)
        #Get the rewards
        reward, done = env_step(action)

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            agent.store_transition(state, action, reward, next_state)

            # Start a new episode
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            agent.store_transition(state, action, reward, next_state)

            # Our state is now the next_state
            state = next_state
        i+=1
    print('\nDone initializing memory')

    #Training
    if training:
        # Initialize the decay rate (that will use to reduce epsilon)

        # Init the game
        game.init()
        episode_rewards = list()
        scores = []
        eps_history = []
        for episode in range(total_episodes):
            total_reward = 0.0
            done = False
            step = 0
            epsilon = agent.epsilon_decay(agent.step_num)
            eps_history.append(epsilon)

            print('Starting episode: ', episode, ', epsilon: %.4f' % epsilon + ' ')

            # Make a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            frames = state
            #frames.shape:(4, 84, 84)

            steps_taken = 0
            print("[Steps]")
            while step < max_steps:
                step += 1

                # Predict the action to take and take it
                action, explore_probability = agent.predict_action(frames)
                # do the action
                reward, done = env_step(possible_actions[action])

                total_reward += reward
                steps_taken = step
                episode_rewards.append(reward)

                sys.stdout.write(f"\r{step}")

                if done:
                    print("\n[DONE]")
                    episode_rewards.append(total_reward)
                    if total_reward > agent.best_reward:
                        agent.best_reward = total_reward

                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Set step = max_steps to end the episode
                    step = max_steps
                    agent.store_transition(state, action, reward, next_state)

                else:
                    # Get the next state and Stack the frame of the next_state
                    next_state = game.get_state().screen_buffer
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    agent.store_transition(state, action, reward, next_state)
                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                agent.learn(batch_size, writer)

            scores.append(total_reward)

            print('\nEpisode finished: {}, '.format(episode),
                  'iterations: {}, '.format(steps_taken),
                  'total reward: {}, '.format(total_reward),
                  'mean reward: {}, '.format(np.mean(episode_rewards)),
                  'best reward: {}, '.format(agent.best_reward),
                  #'Training loss: {:.4f}'.format(brain.loss),
                  #'explore probability: {:.4f}'.format(explore_probability)
                  )
            print("")
            writer.add_scalar("main/ep_reward", total_reward, episode)
            writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), episode)
            writer.add_scalar("main/max_ep_reward", agent.best_reward, episode)

        #Plotting
        x = [i + 1 for i in range(total_episodes)]
        fileName = str(total_episodes) + 'Games' + 'Gamma' + str(agent.gamma) + 'Alpha' + str(agent.lr) + 'Memory' + str(agent.memSize) + '.png'
        plotLearning(x, scores, eps_history, fileName)

        writer.close()
        #tensorboard --logdir=logs/
        #http://localhost:6006/
