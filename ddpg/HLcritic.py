from critic import CriticNetwork

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Input, Lambda, Activation
from keras.layers.merge import concatenate
from keras.initializers import RandomUniform
from keras.models import Sequential, Model


HIDDEN1_UNITS = 100
HIDDEN2_UNITS = 100


class HuberLossCriticNetwork(CriticNetwork):
    def __init__(self, delta_clip, sess, state_size, action_size, gamma, tau, learning_rate):
        self.delta_clip = delta_clip
        super(HuberLossCriticNetwork,self).__init__(sess, state_size, action_size, gamma, tau, learning_rate)

    def huber_loss(self, y_true, y_pred):
        err = y_true - y_pred

        cond = K.abs(err) < self.delta_clip
        L2 = 0.5 * K.square(err)
        L1 = self.delta_clip * (K.abs(err) - 0.5 * self.delta_clip)

        loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

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
