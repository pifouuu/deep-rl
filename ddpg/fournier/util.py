import tensorflow as tf
from gym import wrappers


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def wrap_gym(env,render,dir):
    if not render:
        env = wrappers.Monitor(
            env, dir, video_callable=False, force=True)
    else:
        env = wrappers.Monitor(env, dir, force=True)
    return env


