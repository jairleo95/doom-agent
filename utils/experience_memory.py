from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])

class ExperienceMemory(object):
	"""
	Un buffer que simula la memoria, experiencia del agente
	"""
	def __init__(self, capacity = int(1e6)):
		"""
		:param capacity: Capacidad total de la memoria ciclica (numero maximo de experiencias almacenables)
		:return:
		"""
		self.capacity = capacity
		self.memory_idx = 0 #identificador que sabe la experiencia actual
		self.memory = []

	def sample(self, batch_size):
		"""
		:param que batch_size: Tamano de la memoria a recuperar
		:return: Una muestra aleatoria del tamano batch_size de experiencias de la memoria
		"""
		assert batch_size <= self.get_size(), "El tamano de la muestra es superior a la memoria disponible"
		return random.sample(self.memory, batch_size)

	def get_size(self):
		"""
		:return: Numero de experiencias almacenadas en memoria
		"""
		return len(self.memory)

	def store(self, exp):
		"""
		:param experience: Objecto experiencia a ser almacenado en memoria
		:return:
		"""
		self.memory.insert(self.memory_idx % self.capacity, exp)
		self.memory_idx +=1