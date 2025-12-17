### MODEL HYPERPARAMETERS

state_size = (84, 84, 4)
# state_size = [100, 120, 4]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
learning_rate = 0.00025  # Alpha (aka learning rate)


### TRAINING HYPERPARAMETERS
total_episodes = 1000  # number of games
max_steps = 100
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 5000  # (10000) Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95  # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
pretrain_length = 300  #(30000) Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000  # (100000) Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False