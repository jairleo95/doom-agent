import numpy as np

# class exp_replay():
#     def __init__(self, input_dims, buffer_size= 1000000):
#         self.buffer_size = buffer_size
#         self.state_mem = np.zeros((self.buffer_size, *(input_dims)), dtype=np.float32)
#         self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
#         self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
#         self.next_state_mem = np.zeros((self.buffer_size, *(input_dims)), dtype=np.float32)
#         self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
#         self.pointer = 0
#
#     def add_exp(self, state, action, reward, next_state, done):
#         idx  = self.pointer % self.buffer_size
#         self.state_mem[idx] = state
#         self.action_mem[idx] = action
#         self.reward_mem[idx] = reward
#         self.next_state_mem[idx] = next_state
#         self.done_mem[idx] = 1 - int(done)
#         self.pointer += 1
#
#     def sample_exp(self, batch_size= 64):
#         max_mem = min(self.pointer, self.buffer_size)
#         batch = np.random.choice(max_mem, batch_size, replace=False)
#         states = self.state_mem[batch]
#         actions = self.action_mem[batch]
#         rewards = self.reward_mem[batch]
#         next_states = self.next_state_mem[batch]
#         dones = self.done_mem[batch]
#         return states, actions, rewards, next_states, dones

class ReplayBuffer():
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                    dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
