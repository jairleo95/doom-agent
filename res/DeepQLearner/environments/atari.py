# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 18:58:53 2020

@author: JairL
"""

import gym
import atari_py
import numpy as np
import random
import cv2
from collections import deque
from gym.spaces.box import Box

def get_games_list():
    return atari_py.list_games()

def make_env(env_id, env_conf):
    env = gym.make(env_id)
    if "NoFrameskip" in env_id:
        assert "NoFrameskip" in env.spec.id
        env = NoopResetEnv(env, noop_max = 30)#Ejecutar conjunto aleatorio de acciones en el estado inicial.
        env = MaxAndSkipEnv(env, skip = env_conf['skip_rate'])#establecer cuantas veces se va arepetir
        
    if env_conf['episodic_life']:
        env = EpisodicLifeEnv(env)
        
    try:
        if "FIRE" in env.unweapped.get_action_meanings():
            env = FireResetenv(env)
    except AttributeError:
        pass
    
    env = AtariRescale(env, env_conf['useful_region'])
    
    if env_conf['normalize_observation']:
        env = NormalizedEnv(env)#normalizar las imgagenes entre la media y la varianza
        
    env = FrameStack(env, env_conf['num_frames_to_stack'])
    
    if env_conf['clip_reward']:
        env = CLipReward(env)
        
    return env
    
def process_frame_84(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    #escalar la imagen
    frame *= 1.0/255.0
    
    frame = cv2.resize(frame, (84, conf["dimension2"]))
    frame = cv2.resize(frame, (84,84))
    frame = np.reshape(frame, [1,84,84])
    return frame


#reescalado 0 a 1
class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0, 255, [1, 84, 84], dtype = np.uint8)
        self.conf = env_conf
        
    def observation(self, observation):
        return process_frame_84(observation, self.conf)

    #tipificacion
class NormalizedEnv(gym.ObservationWrapper): 
    def __init__(self, env = None):
        gym.ObservationWrapper.__init__(self, env)
        self.mean = 0
        self.std = 0
        self.alpha = 0.9999
        self.num_steps = 0
        
    def observation(self, observation):
        self.num_steps +=1
        self.mean =self.mean * self.alpha + observation.mean() * (1-self.alpha)
        self.std = self.std * self.alpha + observation.std() * (1-self.alpha)

        unbiased_mean = self.mean /(1-pow(self.alpha, self.num_steps))
        unbiased_std = self.std / (1-pow(self.alpha, self.num_steps))
        
        return (observation - unbiased_mean) /(unbiased_std + 1e-8)#+ 1e-8: para evitar que no se divida entre cero

    #sobreescribir para tener acceso a las funcionalidades del agente
class CLipReward(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def reward(self, reward):
        return np.sign(reward)
    
    #Evitar memorizar la posicion de salida
    #operaciones sin aprendizaje (No operation action = Noop)
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        
        #cuando el agente abre los ojos
    def reset(self):
        self.env.reset()
        noops = random.randrange(1, self.noop_max +1) #numero aleatorio
        assert noops > 0
        observation = None
        
        #saltar las primeras iteraciones con acciones que no hagan nada
        #para evitar que el agente aprenda desde el lugar donde nace sino las acciones
        for _ in range(noops):
            observation, _, done, _ = self.env.step(self.noop_action)
        return observation
    
    def step(self, action):
        return self.env.step(action)
    
    #pulsar el boton fire
class FireResetenv(gym.Wrapper):
   def __init__(self, env):
       gym.Wrapper.__init__(self, env)
       assert env.unwrapped.get_action_meanings()[1] == "FIRE"
       assert len(env.unwrapped.get_action_meanings())>=3

   def reset(self):
       self.env.reset()
       obs, _, done, _ = self.env.step(1)
       
       if done:
           self.env.reset()
       obs, _, done, _ = self.env.step(2)
       
       if done:
           self.env.reset()
    
       return obs
   
   def step(self, action):
        return self.env.step(action)

#para que elk agente aprenda que el juego tiene varias vidas y no solo cuando haya terminado
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.has_really_died = False
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.has_really_died = False
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:
            done = True
            self.has_really_died = True
        self.lives = lives
        
        return obs, reward, done, info
        
    def reset(self):
        if self.has_really_died is False:
            obs = self.env.reset()
            self.lives = 0
            
        else:
            obs,_,_, info = self.env.step(0)
            self.lives = info['ale.lives']
        
        return obs
    
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env = None, skip =4):
        gym.Wrapper.__init__(self, env)
        #encolar lo frames que nose van a procesar
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip
        
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward +=reward
            if done:
                break
        #obtener la mejor de todas las observaciones
        max_frame = np.max(np.stack(self._obs_buffer), axis = 0)
        return max_frame, total_reward, done, info
    
    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
    
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen =k )
        shape = env.observation_space.shape
        self.observation_space = Box(low = 0 ,high = 255,
                                     shape= (shape[0]*k, shape[1], shape[2]), dtype = np.uint8)
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self.get_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.get_obs(), reward, done, info
    
    def get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
class LazyFrames(object):
    def __init__(self, frames):
        self.frames = frames
        self.out = None
     
        #liberar espacio en memoria
        
    def _force(self):
        if self.out is None:
            self.out = np.concatenate(self.frames, axis = 0)
            self.frames = None
        return self.out
    
    def __array_(self, dtype = None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
            
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]
    