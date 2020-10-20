# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 01:18:40 2020

@author: JairL
"""
import torch

class SLP(torch.nn.Module):
    """
    SLP significa Single Layer Perceptron o neurona de una sola capa para aproximar funciones
    """
    
    def __init__(self, input_shape, output_shape, device = torch.device("cpu")):
        """
        :param input_shape: Tamano o forma de los datos de entrada
        :param output_shape: Tamano o forma de los datos de salida
        :param device: El dispositivo ('cpu' o 'cuda') que la SLP debe utilizar para almacenar los inputs a cada iteracion
        """
        
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)
        
    
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x)) #funcion de activacion relu
        x = self.out(x)
        return x