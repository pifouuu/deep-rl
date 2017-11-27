import tensorflow as tf
import numpy as np
import gym
import argparse
import pprint as pp
from logger import Logger
from envWrapper import RandomGoal, NoGoal, HandmadeCurriculum
from memory import Memory, HerMemory
import pickle
import time
import datetime
from actor import ActorNetwork
from critic import CriticNetwork
from ddpgAgent import DDPG_agent
from noise import OrnsteinUhlenbeckActionNoise

#TODO : Update doc on github on this code


def main(args):
    params = '_delta_'+str(args['delta'])+\
              '_wrapper_'+str(args['wrapper'])+\
              '_hindsight_'+str(args['with_hindsight'])
    logdir = args['summary_dir']
    final_dir = logdir+'/'+params+'/'+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    logger_step = Logger(dir=final_dir+'/log_step',format_strs=['json', 'tensorboard'])
    logger_episode = Logger(dir=final_dir+'/log_episodes', format_strs=['stdout', 'json', 'tensorboard'])


    actor_lr = float(args['actor_lr'])
    tau = float(args['tau'])
    critic_lr = float(args['critic_lr'])
    gamma = float(args['gamma'])
    batch_size = int(args['minibatch_size'])
    eval_episodes = int(args['eval_episodes'])
    max_episode_steps = int(args['max_episode_steps'])
    max_steps = int(args['max_steps'])
    eval_freq = int(args['eval_freq'])

    train_env = gym.make(args['env'])
    test_env = gym.make(args['env'])


    if args['wrapper'] == 'NoGoal':
        env_wrapper = NoGoal()
    elif args['wrapper'] == 'RandomGoal':
        env_wrapper = RandomGoal()
    elif args['wrapper'] == 'HandCurri':
        env_wrapper = HandmadeCurriculum()
    else:
        print("Nooooooooooooooooooooo")

    state_dim = env_wrapper.state_shape[0]
    action_dim = env_wrapper.action_shape[0]
    action_bound = train_env.action_space.high
    # Ensure action bound is symmetric
    assert (train_env.action_space.high == -train_env.action_space.low)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    # Initialize replay memory
    if args['with_hindsight']:
        memory = HerMemory(env_wrapper, with_reward=True, limit=int(1e6), strategy='last')
    else:
        memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))


    with tf.Session() as sess:

        if args['random_seed'] is not None:
            np.random.seed(int(args['random_seed']))
            tf.set_random_seed(int(args['random_seed']))
            train_env.seed(int(args['random_seed']))
            test_env.seed(int(args['random_seed']))

        actor = ActorNetwork(sess,
                             state_dim,
                             action_dim,
                             action_bound,
                             tau,
                             actor_lr)

        critic = CriticNetwork(sess,
                               state_dim,
                               action_dim,
                               gamma,
                               tau,
                               critic_lr)

        agent = DDPG_agent(sess,
                           actor,
                           actor_noise,
                           critic,
                           train_env,
                           test_env,
                           env_wrapper,
                           memory,
                           logger_step,
                           logger_episode,
                           batch_size,
                           eval_episodes,
                           max_episode_steps,
                           max_steps,
                           eval_freq)
        agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--delta', help='delta in huber loss', default=None)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)
    parser.add_argument('--wrapper', help='concatenate goal and observation in states', default='NoGoal')
    parser.add_argument('--with-hindsight', help='use hindsight experience replay', action='store_true')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=None)
    parser.add_argument('--max-steps', help='max num of episodes to do while training', default=500000)
    parser.add_argument('--max-episode-steps', help='max number of steps before resetting environment', default=100)
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/v2')
    parser.add_argument('--eval-freq', help='evaluation frequency', default=100)
    parser.add_argument('--eval-episodes', help='number of episodes to run during evaluation', default=20)
    parser.add_argument('--eval-steps', help='number of steps in the environment during evaluation', default=1000)

    parser.set_defaults(with_hindsight=False)

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
