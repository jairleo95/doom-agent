# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:23:16 2020

@author: JairL
"""

import torch
import numpy as np
from libs.perceptron import SLP
from res.DeepQLearner.utils.decay_schedule import LinearDecaySchedule
import random
import gym
from res.DeepQLearner.utils import ExperienceMemory, Experience

MAX_NUM_EPISODES = 100000
STEPS_PER_EPISODE = 300

class ShallowQLearner(object):
    
    def __init__(self, environment, learning_rate = 0.005, gamma = 0.98):
        self.obs_shape = environment.observation_space.shape
              
        self.action_shape = environment.action_space.n
        self.Q = SLP(self.obs_shape, self.action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)

        self.gamma = gamma

        self.epsilon_max = 1.0      
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                        final_value = self.epsilon_min,
                                                        max_steps = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q

        #exp replay
        self.memory = ExperienceMemory(capacity = int(1e5))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
   
    def get_action(self, obs):
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):

        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy())
    
        return action

    def learn(self, obs, action, reward, next_obs):

        #formula
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)

        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()
 
    def replay_experience(self, batch_size):
        """
        Vuelve a jugar usando la experiencia aleatoria almacenada
        :param batch_size: Tamano de la muestra a tomar de la memoria
        :return:
        """
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)

    def learn_from_batch_experience(self, experiences):
        """
        Actualiza la red neuronal profunda en base a lo aprendido en el conjunto de experiencias anteriores
        :param experiences: fragmento de recuerdos anteriores
        :return:
        """
        # el * pasa la referencia y no el valor
        #extraer las obs, acciones, recompensas, las siguientes obs, done
        batch_exp = Experience(*zip(*experiences))
        obs_batch = np.array(batch_exp.obs)
        action_batch = np.array(batch_exp.action)
        reward_batch = np.array(batch_exp.reward)
        next_obs_batch = np.array(batch_exp.next_obs)
        done_batch = np.array(batch_exp.done)#para cada experiencia si hay obs

        #calcular la diferencia temporal, forma de vector
        #~done_batch si es False es cero y multiplica todo la derecha
        td_target = reward_batch + ~done_batch * \
                    np.tile(self.gamma, len(next_obs_batch)) * \
                    self.Q(next_obs_batch).detach().max(1)[0].data.numpy()#convertimos a numpy array para multiplicar y se revierte con torch.from_numpy

        #calcular el error cuadrado medio
        td_target = torch.from_numpy(td_target)#se convierte de numpy ha tensor
        td_target = td_target.to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        td_error = torch.nn.functional.mse_loss(
                    self.Q(obs_batch).gather(1, action_idx.view(-1,1).long()),
                        td_target.float().unsqueeze(1))#se agrega .long() para convertir a tipo LongTensor lo que espera como parametro el gather de action_idx.view(-1,1)

        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_optimizer.step()


if __name__ == "__main__":
    environment = gym.make('CartPole-v0')
    agent = ShallowQLearner(environment)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = environment.reset()
        total_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
                #environment.render()
            #realizar accion
            action = agent.get_action(obs)
            next_obs, reward, done, info = environment.step(action)

            #almacenar en memoria para exp replay
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            #aprendizaje
            agent.learn(obs, action, reward, next_obs)

            obs = next_obs
            total_reward += reward

            if done is True:
                if first_episode:
                    max_reward = total_reward
                    first_episode = False

                episode_rewards.append(total_reward)
                if total_reward > max_reward:
                    max_reward = total_reward
                print("Episodio #{} finalizado con {} iteraciones. Recompensa = {}, Recompensa media = {}, Mejor recompensa = {}".format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))

                #volver a jugar con la experiencia
                if agent.memory.get_size() > 100: #100 de iteraciones
                    agent.replay_experience(32)#colocar 32 experiencias previas aleatorias 
                break
    environment.close()