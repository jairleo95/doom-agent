import tensorflow as tf

#https://pylessons.com/CartPole-DDDQN
#https://github.com/pythonlessons/Reinforcement_Learning/blob/master/03_CartPole-reinforcement-learning_Dueling_DDQN/Cartpole_Double_DDQN_TF2.py

from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, Lambda, Add, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

def DQNModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input

    X = Conv2D(32, 8, strides=(4, 4),activation="relu", padding="valid", input_shape=input_shape,  data_format="channels_last")(X)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_last")(X)
    X = Conv2D(128, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_last")(X)
    X = Flatten()(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
      state_value = Dense(1, kernel_initializer='he_uniform')(X)
      state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

      action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
      action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
        action_advantage)

      X = Add()([state_value, action_advantage])
    else:
      # Output Layer with # of actions: n nodes (left, right, ...)
      X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X)
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model
