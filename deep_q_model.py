import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from res.DeepQLearner.utils.decay_schedule import LinearDecaySchedule

class DeepQNetwork(nn.Module):
    def __init__(self, device):
        super(DeepQNetwork, self).__init__()
        self.device = device
        #padding "valid" is 0
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        # ((84−8+2(0))/4)+1=20
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # ((20−4+2(0))/2)+1=9
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        # ((9−4+2(0))/2)+1=3.5
        self.conv3_bn = nn.BatchNorm2d(128)

        #flatten
        self.flatten = nn.Linear(3*3*128, 512)

        self.fc2 = nn.Linear(512, 3)

        self.loss = nn.MSELoss()


    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
        ##observation = observation.view(-1, 1, 84, 84)
        observation = observation.view(-1, 4, 84, 84)
        observation = F.relu(self.conv1_bn(self.conv1(observation)))
        observation = F.relu(self.conv2_bn(self.conv2(observation)))
        observation = F.relu(self.conv3_bn(self.conv3(observation)))

        observation = observation.view(-1, 3*3*128)
        observation = F.relu(self.flatten(observation))
        actions = self.fc2(observation)
        #print("actions.shape:"+str(actions.shape))
        return actions

class Agent(object):
    def __init__(self, params, maxMemorySize, actionSpace=[0,1,2], writer=None):
        self.params = params

        self.gamma = self.params['gamma']
        self.lr = self.params['learning_rate']

        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")

        self.training_steps_completed = 0
        self.action_shape = actionSpace
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.Q = DeepQNetwork(self.device).to(self.device)
        self.Q_target = DeepQNetwork(self.device).to(self.device)
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        # epsilon greedy
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params['epsilon_max']
        self.epsilon_min = self.params['epsilon_min']
        self.eps_max_steps = self.params['epsilon_decay_final_step']
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps=self.eps_max_steps)

        self.step_num = 0

        #memory
        self.actionSpace = actionSpace#todo: check this actionSpace
        self.memSize = maxMemorySize
        self.memory = []
        self.memCntr = 0

        self.writer = writer


        #Exp replay
        #self.memory = ExperienceMemory(capacity=int(self.params['experience_memory_size']))

    def predict_action(self, state):
        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        #explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        explore_probability = 0
        return self.policy(state), explore_probability

    def epsilon_greedy_Q(self, state):
        self.writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)

        self.step_num += 1  #Para saber en que iteraciones voy
        #print("step_num: "+str(self.step_num))

        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            #print("Take a random step")
            action = np.random.choice(self.actionSpace)
        else:
            #print("Take Not random step")
            ##action = T.argmax(self.Q.forward(state)[1]).item()
            action = T.argmax(self.Q.forward(state)).item()

        return action

    def getMem(self, batch_size):

        if self.memCntr+batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch = self.memory[memStart:memStart+batch_size]

        return np.array(miniBatch)

    def learn(self, batch_size, writer):
        if self.step_num % self.params['target_network_update_frequency'] == 0:
            #copiar pesos
            self.Q_target.load_state_dict(self.Q.state_dict())

        memory = self.getMem(batch_size)

        #Memory columns [state, action, reward, state_]
        Qpred = self.Q.forward(list(memory[:,0][:]))#.to(self.device)
        Qnext = self.Q_target.forward(list(memory[:,3][:]))#.to(self.device)

        #index
        maxA = T.argmax(Qnext, dim=1).to(self.device)
        action_idx = np.arange(batch_size)
        #rewards
        rewards = T.Tensor(list(memory[:, 2])).to(self.device)

        #calculate Q target (td_target = Qtarget)
        #td= Temporal difference
        #TD usan el error o diferencia entre predicciones sucesivas (en lugar del error entre la predicción y la salida final o final del episodio)
        # aprendiendo al existir cambios entre predicciones sucesivas.
        #https://www.udemy.com/course/tensorflow2/learn/lecture/15692530#overview
        td_target = Qpred.clone()
        td_target[action_idx, maxA] = rewards + self.gamma*T.max(Qnext[1])

        td_error = self.Q.loss(td_target, Qpred).to(self.device)

        self.Q_optimizer.zero_grad()
        td_error.backward()

        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def save_model(self):
        file_name = "/tensorboard/dqn/1"
        T.save({'Q': self.Q.state_dict()}, file_name)
        print("Estado del agente guardado en :", file_name)

    def load(self):
        file_name = "/tensorboard/dqn/1"
        agent_state = T.load(file_name, map_location=lambda storage, loc: storage)
        self.Q.load_state_dict(agent_state["Q"])
        self.Q.to(self.device)

        print("El modelo cargado Q desde", file_name,
              "que hasta el momento tiene una mejor recompensa media de: ", self.best_mean_reward,
              "y una recompensa maxima de:", self.best_reward)




