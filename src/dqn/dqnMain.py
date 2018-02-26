from __future__ import division

import tensorflow as tf
import numpy as np
from gym.envs.registration import make
import argparse
import pprint as pp
from ddpg.logger import Logger
from ddpg.memory import SARSTMemory, EpisodicHerSARSTMemory
import datetime
from ddpg.networks import ActorNetwork, HuberLossCriticNetwork
from ddpg.ddpgAgent import DDPG_agent
from ddpg.noise import OrnsteinUhlenbeckActionNoise
import random as rn
import os
from ddpg.util import load, boolean_flag

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from dqn.networks import CriticNetwork
from dqn.dqnAgent import DQN_Agent

from dqn.gridworld import gameEnv

def main(args):
    # Storing logger output in files with names corresponding to parameters used
    params = args['memory'] + '_' + \
             args['strategy'] + '_' + \
             args['sampler'] + '_' + \
             args['alpha'] + '_' + \
             args['delta'] + '_' + \
             args['activation'] + '_' + \
             args['invert_grads'] + '_' + \
             args['target_clip'] + '_' + \
             args['sigma']
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Two loggers are defined to retrieve information by step or by episode. Only episodic information is displayed to stdout.
    log_dir = args['log_dir'] + params + '/' + now
    logger_step = Logger(dir=log_dir + '/log_steps', format_strs=['stdout', 'json'])
    logger_episode = Logger(dir=log_dir + '/log_episodes', format_strs=['stdout', 'json'])

    os.environ['PYTHONHASHSEED'] = '0'
    if args['random_seed'] is not None:
        np.random.seed(int(args['random_seed']))
        rn.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))

    train_env = gameEnv(partial=False,size=84)
    test_env = gameEnv(partial=False,size=84)

    memory = SARSTMemory(train_env, limit=int(1e4))

    with tf.Session() as sess:
        critic = CriticNetwork(sess,
                               [train_env.sizeX, train_env.sizeY, 3],
                               4,
                               .99,
                               0.001,
                               0.0001,
                               512)

        agent = DQN_Agent(sess,
                          critic,
                          train_env,
                          test_env,
                          memory,
                          logger_step,
                          logger_episode,
                          int(args['minibatch_size']),
                          int(args['nb_test_steps']),
                          int(args['max_steps']),
                          log_dir,
                          int(args['save_freq']),
                          args['target_clip'] == 'True',
                          float(args['alpha']),
                          args['render_test'],
                          int(args['train_freq']),
                          int(args['nb_train_iter']),
                          args['resume_step'],
                          args['resume_timestamp'],
                          1,
                          0.1)
        agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    parser.add_argument('--memory', help='type of memory to use', default='sarst')
    parser.add_argument('--strategy', help='hindsight strategy: final, episode or future', default='final')
    parser.add_argument('--sampler', help='type of goal_wrappers sampling', default='no')
    parser.add_argument('--alpha', help="how much priorization in goal_wrappers sampling", default=0.5)
    parser.add_argument('--sigma', help="amount of exploration", default=0.3)
    parser.add_argument('--delta', help='delta in huber loss', default='inf')
    parser.add_argument('--activation', help='actor final layer activation', default='tanh')
    parser.add_argument('--invert-grads', help='Gradient inverting for bounded action spaces', default=False)
    parser.add_argument('--target-clip', help='Reproduce target clipping from her paper', default=False)

    # run parameters
    parser.add_argument('--env', help='choose the gym env', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=None)
    parser.add_argument('--max-steps', help='max num of episodes to do while training', default=200000)
    parser.add_argument('--log-dir', help='directory for storing run info',
                        default='/home/pierre/PycharmProjects/deep-rl/log/local/')
    parser.add_argument('--resume-timestamp', help='directory to retrieve weights of actor and critic',
                        default=None)
    parser.add_argument('--resume-step', help='resume_step',
                        default=None)
    parser.add_argument('--train-freq', help='training frequency', default=100)
    parser.add_argument('--nb-train-iter', help='training iteration number', default=50)
    parser.add_argument('--nb-test-steps', help='number of steps in the environment during evaluation', default=201)
    boolean_flag(parser, 'render-test', default=False)
    parser.add_argument('--save-freq', help='saving models weights frequency', default=50)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)