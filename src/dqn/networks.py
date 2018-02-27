import numpy as np
import math
from keras.initializers import RandomUniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import *
# from keras.layers import Dense, Input, Multiply, Add, Subtract, Conv2D, Lambda, Flatten, RepeatVector
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import os
import json_tricks
from keras.layers.merge import concatenate
from keras.utils import plot_model


# Generic deep rl network class with generic functionalities
class Network(object):
    def __init__(self, sess, state_dim, action_dim, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        K.set_session(sess)
        self.model = None
        self.t_model = None

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.t_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.t_model.set_weights(target_weights)

    def hard_target_update(self):
        self.t_model.set_weights(self.model.get_weights())

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def save_target_weights(self, filepath, overwrite=False):
        self.t_model.save_weights(filepath, overwrite=overwrite)

    def save_model(self, filepath, overwrite=False):
        self.model.save(filepath, overwrite=overwrite)

    def save_target_model(self, filepath, overwrite=False):
        self.t_model.save(filepath, overwrite=overwrite)

    def save_archi(self, filepath, overwrite=False):
        model_json = self.model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(json_tricks.dumps(json_tricks.loads(model_json), indent=4))

    def save_target_archi(self, filepath, overwrite=False):
        target_model_json = self.t_model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(json_tricks.dumps(json_tricks.loads(target_model_json), indent=4))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def load_target_weights(self, filepath):
        self.t_model.load_weights(filepath)

    def print_target_weights(self):
        print (self.t_model.get_weights())

    def print_weights(self):
        print (self.model.get_weights())

class CriticNetwork(Network):
    def __init__(self, sess, state_size, action_size, gamma, tau, learning_rate, depths, kernels, strides):
        super(CriticNetwork, self).__init__(sess, state_size, action_size, tau, learning_rate)
        self.gamma = gamma
        self.depths = depths
        self.kernels = kernels
        self.strides = strides

        self.model, self.states, self.actions, self.q_values = self.create_critic_network()
        self.t_model, self.t_states, self.t_actions, self.t_q_values = self.create_critic_network()

        self.select_action = K.argmax(self.q_values, axis=1)

    def create_critic_network(self):

        input_state = Input(shape=self.s_dim)
        conv = input_state
        for d, k, s in zip(self.depths, self.kernels, self.strides):
            conv = Conv2D(d, kernel_size=(k,k), strides=(s,s), activation="relu")(conv)

        # conv1 = Conv2D(32, kernel_size=(8,8), strides=(4,4), activation="relu")(input_state)
        # conv2 = Conv2D(64, kernel_size=(4,4), strides=(2,2), activation="relu")(conv1)
        # conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(conv2)
        # x = Conv2D(h_dim, kernel_size=(7, 7), strides=(1, 1), activation="relu")(conv3)
        adv = Lambda(lambda x: x[:,:,:,:int(self.depths[-1] / 2)], name='First_half')(conv)
        val = Lambda(lambda x: x[:, :, :,int(self.depths[-1] / 2):], name='Second_half')(conv)
        adv = Flatten()(adv)
        val = Flatten()(val)
        adv = Dense(self.a_dim[0], activation='linear',
                    kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(adv)
        mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name='Mean')(adv)
        mean = RepeatVector(self.a_dim[0])(mean)
        mean = Flatten()(mean)
        sub = Subtract()([adv, mean])
        val = Dense(1, activation='relu')(val)
        val = RepeatVector(self.a_dim[0])(val)
        val = Flatten()(val)
        q_values = Add()([val, sub])

        input_action = Input(shape=self.a_dim, dtype='int32')
        action_one_hot = Lambda(lambda x: K.one_hot(x, self.a_dim[0]), output_shape=lambda s: (s[0], self.a_dim[0]), name='One_hot')(input_action)
        mul = Multiply()([q_values, action_one_hot])
        output = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0],1), name='Sum')(mul)

        model = Model(inputs=[input_state, input_action], outputs=output)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model, input_state, input_action, q_values

    def predict_target_action_values(self, states):
        return self.sess.run(self.t_q_values, feed_dict={self.t_states: states})

    def select_actions(self, states):
        return self.sess.run(self.select_action, feed_dict={self.states: states})

    def train(self, states, actions, targets):
        self.model.train_on_batch([states, actions], targets)


