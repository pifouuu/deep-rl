import numpy as np
import math
from keras.initializers import RandomUniform
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Lambda, Activation
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN1_UNITS = 100
HIDDEN2_UNITS = 100


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, gamma, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.gamma = gamma
        self.s_dim = state_size
        self.a_dim = action_size
        self.learning_rate = learning_rate
        self.action_size = action_size

        self.stat_ops = []
        self.stat_names = []

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.out = self.model.output
        self.action_grads = tf.gradients(self.out, self.action)  # GRADIENTS for policy update

        # Setting up stats
        self.stat_ops += [tf.reduce_mean(self.out)]
        self.stat_names += ['Mean Q values']
        self.stat_ops += [tf.reduce_mean(self.action_grads)]
        self.stat_names += ['reference_action_grads']

        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w = Dense(400, activation="relu", kernel_initializer="he_uniform")(S)
        h = concatenate([w, A])
        h3 = Dense(300, activation="relu", kernel_initializer="he_uniform")(h)
        V = Dense(1, activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S

    def get_stats(self, stats_sample):
        critic_values = self.sess.run(self.stat_ops, feed_dict={
            self.state: stats_sample['state0'],
            self.action: stats_sample['action'],
        })

        names = self.stat_names[:]
        assert len(names) == len(critic_values)
        stats = dict(zip(names, critic_values))

        # critic_with_actor_values = self.sess.run(self.stats_ops, feed_dict={
        #     self.inputs: stats_sample[0],
        #     self.action: stats_sample['action'],
        # })
        #
        # for name, val in zip(names, critic_with_actor_values):
        #     stats[name+'_actor'] = val

        return stats