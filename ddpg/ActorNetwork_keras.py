import numpy as np
import math
from keras.initializers import RandomUniform
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, action_bound, tau, learning_rate):
        self.sess = sess
        self.tau = tau
        self.s_dim = state_size
        self.a_dim = action_size
        self.learning_rate = learning_rate
        self.action_bound = action_bound

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

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
        V = Dense(action_dim, activation="tanh",
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h1)
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S
