#!/usr/bin/env python
from __future__ import print_function


import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv2D, Dense, Flatten,  MaxPooling2D, Input, AveragePooling2D, Lambda,  Activation, Embedding
from keras.optimizers import SGD, Adam
from keras.layers import LSTM, GRU, BatchNormalization
from keras import backend as K

import tensorflow as tf
#tf.python.control_flow_ops = tf

class Networks(object):

    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution

        With States as inputs and output Probability Distributions for all Actions
        """

        state_input = Input(shape=(input_shape))
        cnn_feature = Conv2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
        cnn_feature = Conv2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
        cnn_feature = Conv2D(64, 3, 3, activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    @staticmethod
    def actor_network(input_shape, action_size, learning_rate):
        """Actor Network for A2C
        """

        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 4, strides=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(units=64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(units=32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(units=action_size, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    @staticmethod
    def critic_network(input_shape, value_size, learning_rate):
        """Critic Network for A2C
        """

        model = Sequential()
        model.add(Conv2D(32, 8, strides=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 4, strides=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(units=64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(units=32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(units=value_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod
    def policy_reinforce(input_shape, action_size, learning_rate):
        """
        Model for REINFORCE
        """

        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=action_size, activation='softmax'))

        adam = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    @staticmethod
    def dqn(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(Conv2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input_shape)))
        model.add(Conv2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod
    def drqn(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, 8, 8, subsample=(4,4), activation='relu'), input_shape=(input_shape)))
        model.add(TimeDistributed(Conv2D(64, 4, 4, subsample=(2,2), activation='relu')))
        model.add(TimeDistributed(Conv2D(64, 3, 3, activation='relu')))
        model.add(TimeDistributed(Flatten()))

        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        model.add(LSTM(512,  activation='tanh'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod
    def a2c_lstm(input_shape, action_size, value_size, learning_rate):
        """Actor and Critic Network share convolution layers with LSTM
        """

        state_input = Input(shape=(input_shape)) # 4x64x64x3
        x = TimeDistributed(Conv2D(32, 8, 8, subsample=(4,4), activation='relu'))(state_input)
        x = TimeDistributed(Conv2D(64, 4, 4, subsample=(2,2), activation='relu'))(x)
        x = TimeDistributed(Conv2D(64, 3, 3, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)

        x = LSTM(512, activation='tanh')(x)

        # Actor Stream
        actor = Dense(action_size, activation='softmax')(x)

        # Critic Stream
        critic = Dense(value_size, activation='linear')(x)

        model = Model(input=state_input, output=[actor, critic])

        adam = Adam(lr=learning_rate, clipnorm=1.0)
        model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=adam, loss_weights=[1., 1.])

        return model
