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

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, action_bound, tau, learning_rate, activation):
        self.sess = sess
        self.tau = tau
        self.s_dim = state_size
        self.a_dim = action_size
        self.learning_rate = learning_rate
        self.action_bound = action_bound
        self.stat_ops = []
        self.stat_names = []
        self.activation = activation

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
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

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

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
