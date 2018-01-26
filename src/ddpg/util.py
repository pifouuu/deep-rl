import tensorflow as tf
from gym import wrappers
import pkg_resources


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

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result

def boolean_flag(parser, name, default=False, help=None):
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser: argparse.Parser
        parser to add the flag to
    name: str
        --<name> will enable the flag, while --no-<name> will disable it
    default: bool or None
        default value of the flag
    help: str
        help string for the flag
    """
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)