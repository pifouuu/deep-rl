import numpy as np
import math
from keras.initializers import RandomUniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import os
import json_tricks
from keras.layers.merge import concatenate


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

class ActorNetwork(Network):
    def __init__(self, sess, state_size, action_size, tau, learning_rate, activation):
        super(ActorNetwork, self).__init__(sess, state_size, action_size, tau, learning_rate)
        self.activation = activation

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(self.s_dim, self.a_dim)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.s_dim, self.a_dim)
        self.action_gradient = tf.placeholder(tf.float32,[None, self.a_dim])
        self.out = self.model.output
        self.params_grad = tf.gradients(self.out, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

        self.stat_ops += [tf.reduce_mean(self.out)]
        self.stat_names += ["mean_action"]
        self.stat_ops += [tf.reduce_mean(self.target_model.output)]
        self.stat_names += ["mean_target_action"]

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def predict_target(self, states):
        return self.target_model.predict_on_batch(states)

    def predict(self, states):
        return self.model.predict_on_batch(states)

    def create_actor_network(self, state_size,action_dim):
        S = Input(shape=[state_size])
        h0 = Dense(400, activation="relu", kernel_initializer="he_uniform")(S)
        h1 = Dense(300, activation="relu", kernel_initializer="he_uniform")(h0)
        V = Dense(action_dim, activation=self.activation,
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h1)
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S

    def get_stats(self, stats_sample):
        actor_values = self.sess.run(self.stat_ops, feed_dict={
            self.state: stats_sample['state0'],
            self.target_state: stats_sample['state0']
        })

        names = self.stat_names[:]
        assert len(names) == len(actor_values)
        stats = dict(zip(names, actor_values))

        return stats

class CriticNetwork(Network):
    def __init__(self, sess, state_size, action_size, gamma, tau, learning_rate):
        super(CriticNetwork, self).__init__(sess, state_size, action_size, tau, learning_rate)
        self.gamma = gamma

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(self.s_dim, self.a_dim)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.s_dim, self.a_dim)
        self.out = self.model.output
        self.action_grads = tf.gradients(self.out, self.action)  # GRADIENTS for policy update

        # Setting up stats
        self.stat_ops += [tf.reduce_mean(self.out)]
        self.stat_names += ['mean_Q_values']
        self.stat_ops += [tf.reduce_mean(self.target_model.output)]
        self.stat_names += ['mean_target_Q_values']
        self.stat_ops += [tf.reduce_mean(self.action_grads)]
        self.stat_names += ['reference_action_grads']

    def gradients(self, states, actions):
        out, grads =  self.sess.run([self.out, self.action_grads], feed_dict={
            self.state: states,
            self.action: actions
        })
        return out, grads[0]

    def predict_target(self, states, actions):
        return self.target_model.predict_on_batch([states, actions])

    def predict(self, states, actions):
        return self.model.predict_on_batch([states, actions])

    def train(self, states, actions, targets):
        return self.model.train_on_batch([states, actions], targets)

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
            self.target_state: stats_sample['state0'],
            self.target_action: stats_sample['action'],
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
        w = Dense(400, activation="relu", kernel_initializer="he_uniform")(S)
        h = concatenate([w, A])
        h3 = Dense(300, activation="relu", kernel_initializer="he_uniform")(h)
        V = Dense(1, activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss=self.huber_loss, optimizer=adam)
        return model, A, S
