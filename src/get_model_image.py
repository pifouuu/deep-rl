from keras.utils import plot_model

from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.models import Model

from keras.layers import *

from keras.layers import Dense

input_state = Input(shape=(84,84,3))
conv = input_state
for d, k, s in zip([32,64,64,128],[8,4,3,7],[4,2,1,1]):
    conv = Conv2D(d, kernel_size=(k,k), strides=(s,s), activation="relu")(conv)

# conv1 = Conv2D(32, kernel_size=(8,8), strides=(4,4), activation="relu")(input_state)
# conv2 = Conv2D(64, kernel_size=(4,4), strides=(2,2), activation="relu")(conv1)
# conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(conv2)
# x = Conv2D(h_dim, kernel_size=(7, 7), strides=(1, 1), activation="relu")(conv3)
adv = Lambda(lambda x: x[:,:,:,:int(128 / 2)], name='First_half')(conv)
val = Lambda(lambda x: x[:, :, :,int(128 / 2):], name='Second_half')(conv)
adv = Flatten()(adv)
val = Flatten()(val)
adv = Dense(4, activation='linear',
            kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(adv)
mean = Lambda(lambda x: K.mean(x, axis=1, keepdims=True), name='Mean')(adv)
mean = RepeatVector(4)(mean)
mean = Flatten()(mean)
sub = Subtract()([adv, mean])
val = Dense(1, activation='relu')(val)
val = RepeatVector(4)(val)
val = Flatten()(val)
q_values = Add()([val, sub])

input_action = Input(shape=[1], dtype='int32')
action_one_hot = Lambda(lambda x: K.one_hot(x, 4), output_shape=lambda s: (s[0], 4), name='One_hot')(input_action)
mul = Multiply()([q_values, action_one_hot])
output = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0],1), name='Sum')(mul)

model = Model(inputs=[input_state, input_action], outputs=output)
model.compile(loss='mse', optimizer=Adam(lr=0.1))
plot_model(model, show_shapes=True)
