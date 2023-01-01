import torch as T
import torch.optim as optim
import numpy as np
import random
from utils.linear_decay_schedule import LinearDecaySchedule
from dqn.deep_q_model import DeepQNetwork

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

        #Q_target son los valores conocidos (del step anterior)
        self.Q_target = DeepQNetwork(self.device).to(self.device)
        #Q son los valores actuales
        self.Q = DeepQNetwork(self.device).to(self.device)
        #funcion de perdida
        self.Q_optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

        # epsilon greedy: this policy helps as exploration(be able to detect others results by taking more Qs)
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = self.params['epsilon_max']
        self.epsilon_min = self.params['epsilon_min']
        self.eps_max_steps = self.params['epsilon_decay_final_step']
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max, final_value=self.epsilon_min, max_steps=self.eps_max_steps)

        self.step_num = 0

        #memory
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

    # def predict_action_2(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    #     ## EPSILON GREEDY STRATEGY
    #     # Choose action a from state s using epsilon greedy.
    #     ## First we randomize a number
    #     exp_exp_tradeoff = np.random.rand()
    #
    #     # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    #     explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    #
    #     if (explore_probability > exp_exp_tradeoff):
    #         # Make a random action (exploration)
    #         action = random.choice(possible_actions)
    #
    #     else:
    #         # Get action from Q-network (exploitation)
    #         # Estimate the Qs values state
    #         Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
    #
    #         # Take the biggest Q value (= the best action)
    #         choice = np.argmax(Qs)
    #         action = possible_actions[int(choice)]
    #
    #     return action, explore_probability

    def epsilon_greedy_Q(self, state):
        self.writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)

        self.step_num += 1  #Para saber en que iteraciones voy
        #print("step_num: "+str(self.step_num))

        if random.random() < self.epsilon_decay(self.step_num) and not self.params["test"]:
            #print("Take a random step")
            action = np.random.choice(self.action_shape)
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
        print("states before train:", memory[:,0][:].shape)
        #states before train: (32,)
        #torch.Size([32, 4, 84, 84])
        #Q(t) q_eval=q_pred
        q_eval = self.Q_target.forward(list(memory[:,0][:]))#.to(self.device)
        #Q(t-1) q_next_state
        q_next = self.Q.forward(list(memory[:,3][:]))#.to(self.device)

        #index
        max_actions = T.argmax(q_next, dim=1).to(self.device)
        batch_index = np.arange(batch_size)
        #rewards
        rewards = T.Tensor(list(memory[:, 2])).to(self.device)

        #calculate Q target (td_target = Qtarget)
        #td= Temporal difference
        #TD usan el error o diferencia entre predicciones sucesivas (en lugar del error entre la predicción y la salida final o final del episodio)
        # aprendiendo al existir cambios entre predicciones sucesivas.
        #https://www.udemy.com/course/tensorflow2/learn/lecture/15692530#overview
        #T.max(Qnext[1]) permite obtener el mejor Q para la accion actual (Qnext)
        td_target = q_eval.clone()
        td_target[batch_index, max_actions] = rewards + self.gamma*T.max(q_next[1])

        #calcular perdida (backpropagation= ajusta pesos de la RN)
        td_error = self.Q.loss(td_target, q_eval).to(self.device)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def store_transition(self, state, action, reward, state_):
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