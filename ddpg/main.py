import tensorflow as tf
import numpy as np
import gym
import argparse
import pprint as pp
from logger import Logger
from envWrapper import WithGoal, NoGoalWrapper, GoalCurriculum, IntervalCurriculum
from memory import SASMemory, EpisodicHerSASMemory, SARSTMemory, EpisodicHerSARSTMemory
import pickle
import time
import datetime
from actor import ActorNetwork
from HLcritic import HuberLossCriticNetwork
from critic import CriticNetwork
from ddpgAgent import DDPG_agent
from noise import OrnsteinUhlenbeckActionNoise


#TODO : Update doc on github on this code


def main(args):

    params = args['memory'] +'_'+\
        args['strategy'] +'_'+\
        args['sampler'] +'_'+\
        args['alpha'] +'_'+\
        args['delta'] +'_'+\
        args['activation'] +'_'+\
        args['invert_grads'] +'_'+\
        args['target_clip'] +'_'+\
        args['max_episode_steps']

    print(params)

    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    final_dir = args['summary_dir']+params+'/'+now
    save_dir = args['save_dir']+params+'/'+now

    logger_step = Logger(dir=final_dir+'/log_steps', format_strs=['json'])
    logger_episode = Logger(dir=final_dir+'/log_episodes', format_strs=['stdout', 'json'])

    train_env = gym.make(args['env'])
    test_env = gym.make(args['env'])

    env_wrapper = None
    if args['sampler'] == 'no':
        env_wrapper = NoGoalWrapper()
    elif args['sampler'] == 'rnd':
        env_wrapper = WithGoal()
    elif args['sampler'] == 'init':
        env_wrapper = WithGoal()
    elif args['sampler'] == 'intervalC':
        env_wrapper = IntervalCurriculum()
    elif args['sampler'] == 'goalC':
        env_wrapper = GoalCurriculum()
    elif args['sampler'] == 'comp':
        env_wrapper = GoalCurriculum()
    else:
        print("Nooooooo")

    memory = None
    if args['memory'] == 'sas':
        memory = SASMemory(env_wrapper, limit=int(1e6))
    elif args['memory'] == 'sarst':
        memory = SARSTMemory(env_wrapper, limit=int(1e6))
    elif args['memory'] == 'hsarst':
        memory = EpisodicHerSARSTMemory(env_wrapper, limit=int(1e6), strategy=args['strategy'])
    elif args['memory'] == 'hsas':
        memory = EpisodicHerSASMemory(env_wrapper, limit=int(1e6), strategy=args['strategy'])
    else:
        print("Nooooooo")

    state_dim = env_wrapper.state_shape[0]
    action_dim = env_wrapper.action_shape[0]
    action_bound = train_env.action_space.high
    # Ensure action bound is symmetric
    assert (train_env.action_space.high == -train_env.action_space.low)

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))


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
                             float(args['tau']),
                             float(args['actor_lr']),
                             args['activation'])

        # actor.load_weights(args['save_dir']+params+'/2017_11_30_16_56_43/actor_weights_300.h5')

        critic = HuberLossCriticNetwork(sess,
                               state_dim,
                               action_dim,
                               float(args['delta']),
                               float(args['gamma']),
                               float(args['tau']),
                               float(args['critic_lr']))

        # critic = CriticNetwork(sess,
        #                                 state_dim,
        #                                 action_dim,
        #                                 float(args['gamma']),
        #                                 float(args['tau']),
        #                                 float(args['critic_lr']))

        agent = DDPG_agent(sess,
                           actor,
                           actor_noise,
                           critic,
                           train_env,
                           test_env,
                           env_wrapper,
                           memory,
                           args['sampler'],
                           logger_step,
                           logger_episode,
                           int(args['minibatch_size']),
                           int(args['eval_episodes']),
                           int(args['max_episode_steps']),
                           int(args['max_steps']),
                           int(args['eval_freq']),
                           save_dir,
                           int(args['save_freq']),
                           int(args['log_freq']),
                           args['target_clip']=='True',
                           args['invert_grads']=='True',
                           float(args['alpha']))
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
    parser.add_argument('--sampler', help='type of goal sampling', default='no')
    parser.add_argument('--alpha', help="how much priorization in goal sampling", default=0.5)
    parser.add_argument('--delta', help='delta in huber loss', default='inf')
    parser.add_argument('--activation', help='actor final layer activation', default='tanh')
    parser.add_argument('--invert-grads', help='Gradient inverting for bounded action spaces', default=False)
    parser.add_argument('--target-clip', help='Reproduce target clipping from her paper', default=False)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=None)
    parser.add_argument('--max-steps', help='max num of episodes to do while training', default=200000)
    parser.add_argument('--max-episode-steps', help='max number of steps before resetting environment', default=1000)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default='/home/pierre/PycharmProjects/deep-rl/ddpg/results/')
    parser.add_argument('--save-dir', help='directory to store weights of actor and critic',
                        default='/home/pierre/PycharmProjects/deep-rl/ddpg/saves/')
    parser.add_argument('--eval-freq', help='evaluation frequency', default=1000)
    parser.add_argument('--save-freq', help='saving models weights frequency', default=10000)
    parser.add_argument('--log-freq', help='saving models weights frequency', default=200)
    parser.add_argument('--eval-episodes', help='number of episodes to run during evaluation', default=10)
    parser.add_argument('--eval-steps', help='number of steps in the environment during evaluation', default=200)


    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
