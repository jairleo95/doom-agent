import os
import tensorflow.keras as keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import RMSprop

def A2CModel(input_shape, action_space, lr):
    #Advantage Actor-Critic
    X_input = Input(input_shape)
    print("input_shape: ", input_shape)

    X = Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")(X_input)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(128, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Flatten()(X)

    # X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs=X_input, outputs=action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs=X_input, outputs=value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    Actor.summary()
    Critic.summary()

    return Actor, Critic