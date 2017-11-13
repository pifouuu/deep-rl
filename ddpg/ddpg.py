""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
import tflearn
import argparse
import pprint as pp
import logger
from env_wrapper import GoalContinuousMCWrapper, ContinuousMCWrapper
from memory import Memory, HerMemory
import os
import pickle
import time
import datetime
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from training import train
from util import wrap_gym


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

# def build_summaries():
#     episode_reward = tf.Variable(0.)
#     tf.summary.scalar("Reward", episode_reward)
#     episode_ave_max_q = tf.Variable(0.)
#     tf.summary.scalar("Qmax Value", episode_ave_max_q)
#
#     summary_vars = [episode_reward, episode_ave_max_q]
#     summary_ops = tf.summary.merge_all()
#
#     return summary_ops, summary_vars


def main(args):
    dirname = '_tau_'+str(args['tau'])+'_batchsize_'+str(args['minibatch_size'])+'_goal_'+str(args['with_goal'])+\
              '_hindsight_'+str(args['with_hindsight'])+'_eval_'+str(args['eval'])
    dir = args['summary_dir']+dirname
    dir = dir+'_'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logger.configure(dir=dir,format_strs=['stdout', 'json', 'tensorboard'])
    #logger.configure(dir=args['summary_dir'],format_strs=['stdout'])


    with tf.Session() as sess:

        env = gym.make(args['env'])
        eval_env = None
        if args['eval']:
            eval_env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
        if args['eval']:
            eval_env.seed(int(args['random_seed']))


        if args['with_goal']:
            env_wrapper = GoalContinuousMCWrapper()
        else:
            env_wrapper = ContinuousMCWrapper()

        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.shape[0]
        state_dim = env_wrapper.state_shape[0]
        action_dim = env_wrapper.action_shape[0]

        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']), float(args['delta']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # Initialize replay memory
        if args['with_hindsight']:
            memory = HerMemory(env_wrapper, with_reward=True, limit=int(1e6), strategy='last')
        else:
            memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))

        if args['use_gym_monitor']:
            env = wrap_gym(env, args['render_env'], args['monitor_dir'])
            if eval:
                eval_env = wrap_gym(eval_env, args['render_eval_env'], args['monitor_dir'])


        train(sess, env, eval_env, args, actor, critic, actor_noise, memory, env_wrapper)

        if args['use_gym_monitor']:
            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--delta', help='delta in huber loss', default=1.)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)
    parser.add_argument('--with-goal', help='concatenate goal and observation in states', action='store_true')
    parser.add_argument('--with-hindsight', help='use hindsight experience replay', action='store_true')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=0)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=500)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--render-eval-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')
    parser.add_argument('--eval', help='perform regular evaluation on the main task', action='store_true')
    parser.add_argument('--eval-freq', help='evaluation frequency', default=10)
    parser.add_argument('--eval-steps', help='number of steps in the environment during evaluation', default=1000)


    parser.set_defaults(render_env=False)
    parser.set_defaults(render_eval_env=False)
    parser.set_defaults(use_gym_monitor=False)
    parser.set_defaults(with_goal=True)
    parser.set_defaults(with_hindsight=False)
    parser.set_defaults(eval=False)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
