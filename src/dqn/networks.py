import numpy as np
import math
from keras.initializers import RandomUniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Multiply, Add, Subtract
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import os
import json_tricks
from keras.layers.merge import concatenate

# Generic deep rl network class with generic functionalities
class Network(object):
    def __init__(self, sess, state_size, action_size, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.s_dim = state_size
        self.a_dim = action_size
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        K.set_session(sess)
        self.model = None
        self.target_model = None

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)

    def hard_target_update(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def save_target_weights(self, filepath, overwrite=False):
        self.target_model.save_weights(filepath, overwrite=overwrite)

    def save_model(self, filepath, overwrite=False):
        self.model.save(filepath, overwrite=overwrite)

    def save_target_model(self, filepath, overwrite=False):
        self.target_model.save(filepath, overwrite=overwrite)

    def save_archi(self, filepath, overwrite=False):
        model_json = self.model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(json_tricks.dumps(json_tricks.loads(model_json), indent=4))

    def save_target_archi(self, filepath, overwrite=False):
        target_model_json = self.target_model.to_json()
        with open(filepath, "w") as json_file:
            json_file.write(json_tricks.dumps(json_tricks.loads(target_model_json), indent=4))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def load_target_weights(self, filepath):
        self.target_model.load_weights(filepath)

    def print_target_weights(self):
        print (self.target_model.get_weights())

    def print_weights(self):
        print (self.model.get_weights())

class CriticNetwork(Network):
    def __init__(self, sess, state_size, action_size, gamma, tau, learning_rate, h_dim):
        super(CriticNetwork, self).__init__(sess, state_size, action_size, tau, learning_rate)
        self.gamma = gamma
        self.h_dim = h_dim

        self.action = Input(shape=[action_size])
        self.model, self.state = self.create_critic_network(self.s_dim)
        self.target_model, self.target_state = self.create_critic_network(self.s_dim)
        self.out = self.model.output
        self.target_out = self.target_model.output
        self.action_one_hot = K.one_hot(self.action, self.a_dim)
        self.mul = Multiply()([self.out, self.action_one_hot])
        self.q_values = K.sum(self.mul, axis=1)
        self.select_action = K.argmax(self.out, axis=1)
        # Setting up stats
        self.stat_ops += [tf.reduce_mean(self.out)]
        self.stat_names += ['mean_Q_values']

    def create_critic_network(self, state_size):
        S = Input(shape=[state_size])
        x = Dense(64, activation="relu", kernel_initializer="he_uniform")(S)
        x = Dense(64, activation="relu", kernel_initializer="he_uniform")(x)
        x = Dense(self.h_dim, activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(x)
        adv, val = [x[:self.h_dim / 2], x[self.h_dim / 2:]]
        adv = Dense(self.a_dim, activation='linear',
                    kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(adv)
        mean = K.mean(adv, axis=1, keepdims=True)
        sub = Subtract()([adv, mean])
        val = Dense(1, activation='linear',
                    kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(val)
        self.out = Add()([val, sub])
        model = Model(inputs=self.state, outputs=self.out)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, S

    def predict_target_action_values(self, states):
        return self.target_model.predict_on_batch([states])

    def select_actions(self, states):
        return self.sess.run(self.select_action, feed_dict={self.state:states})

    def train(self, states, actions, targets):
        self.model.train_on_batch([states, actions], targets)

class HuberLossCriticNetwork(CriticNetwork):
    def __init__(self, sess, state_size, action_size, delta_clip, gamma, tau, learning_rate):
        self.delta_clip = delta_clip
        super(HuberLossCriticNetwork,self).__init__(sess, state_size, action_size, gamma, tau, learning_rate)

    def huber_loss(self, y_true, y_pred):
        err = y_true - y_pred
        L2 = 0.5 * K.square(err)

        # Deal separately with infinite delta (=no clipping)
        if np.isinf(self.delta_clip):
            return K.mean(L2)

        cond = K.abs(err) < self.delta_clip
        L1 = self.delta_clip * (K.abs(err) - 0.5 * self.delta_clip)
        loss = tf.where(cond, L2, L1)

        return K.mean(loss)

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w = Dense(64, activation="relu", kernel_initializer="he_uniform")(S)
        h = concatenate([w, A])
        h3 = Dense(64, activation="relu", kernel_initializer="he_uniform")(h)
        V = Dense(1, activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss=self.huber_loss, optimizer=adam)
        return model, A, S
