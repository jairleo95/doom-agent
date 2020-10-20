#https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning
from builtins import print
import torch
from datetime import datetime
from deep_q_model import Agent
import numpy as np
from argparse import ArgumentParser
from collections import deque

#doom
from vizdoom import *
import random
from skimage import transform

from res.DeepQLearner.utils.params_manager import ParamsManager

# tensorboardX
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings('ignore')
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# contador global de ejecuciones
global_step_num = 0

def show_image(frame):
    #img = mpimg.imread(frame)
    imgplot = plt.imshow(frame)
    plt.show()


def create_environment():
    game = DoomGame()
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
        stacked_state = np.stack(stacked_frames, axis=2)
        #https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/
        #change channel ordering
        stacked_state = np.moveaxis(stacked_state, 2, 0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)
        stacked_state = np.moveaxis(stacked_state, 2, 0)
        #show_image(stacked_state)

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
global_step_num = 0

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

    ### MODEL HYPERPARAMETERS
    #state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
    #action_size = game.get_available_buttons_size()  # 3 possible actions: left, right, shoot

    ### TRAINING HYPERPARAMETERS
    #max_steps = agent_params["total_episodes"]  #Max possible steps in an episode
    #memory_size = 5000
    training = True

    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test

    brain = Agent(agent_params,
                  maxMemorySize=agent_params["experience_memory_size"])

    # Render the environment
    game.new_episode()
    i = 0

    print("Filling memory: ")
#    for i in range(pretrain_length):
    while brain.memCntr < brain.memSize:
        sys.stdout.write(f"\r{str(i)}")
        if i == 0:
            # First we need a state
            state = game.get_state().screen_buffer
            #state.shape (120, 160)
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            #show_image(state)

        action = random.choice(possible_actions)
        reward, done = env_step(action)

        if done:
            next_state = np.zeros(state.shape)
            brain.storeTransition(state, action, reward, next_state)

            game.new_episode()

            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        else:
            # Get the next state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            brain.storeTransition(state, action, reward, next_state)
            observation = next_state
        i+=1
    print('')
    print('Done initializing memory')


    #Training
    if training:
        # Init the game
        game.init()
        episode_rewards = list()
        for episode in range(agent_params["total_episodes"]):
            total_reward = 0.0
            done = False
            step = 0

            print('[Starting episode: ',episode,', epsilon: %.4f' % brain.epsilon_decay(brain.step_num) + ' ]')

            #env reset
            game.new_episode()
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            frames = state
            #frames.shape:(4, 84, 84)

            print("Steps taken: ")
            while step < agent_params["max_steps"]:

                #sys.stdout.write(f"\r{str(step)}")

                action, explore_probability = brain.predict_action(frames)
                # do the action
                reward, done = env_step(possible_actions[action])

                total_reward += reward
                step += 1
                episode_rewards.append(reward)

                if done:
                    print("[DONE]")
                    episode_rewards.append(total_reward)

                    if total_reward > brain.best_reward:
                        brain.best_reward = total_reward

                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    #step = agent_params["max_steps"]

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    brain.storeTransition(state, action, reward, next_state)

                else:
                    # Get the next state
                    next_state = game.get_state().screen_buffer

                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    brain.storeTransition(state, action, reward, next_state)

                    # st+1 is now our current state
                    state = next_state

                brain.learn(agent_params['batch_size'])
                lastAction = action
            print('Episode Finished: {}'.format(episode), 'Iterations: {}'.format(step),
                  'Total reward: {}'.format(total_reward),
                  # 'Training loss: {:.4f}'.format(brain.loss),
                  'Explore P: {:.4f}'.format(explore_probability)
                  )

    ##Test


