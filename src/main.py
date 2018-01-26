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

def main(args):
    """Despite following the directives of https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development, fully reproducible results could not be obtained. See here : https://github.com/keras-team/keras/issues/2280 for any improvements"""

    os.environ['PYTHONHASHSEED'] = '0'
    if args['random_seed'] is not None:
        np.random.seed(int(args['random_seed']))
        rn.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))

    # Make calls EnvRegistry.make, which builds the environment from its specs defined in gym.envs.init end then builds a timeLimit wrapper around the environment to set the max amount of steps to run
    train_env = make(args['env'])
    test_env = make(args['env'])

    # Wraps each environment in a goal_wrapper to override basic env methods and be able to access goal space properties, or modify the environment simulation according to sampled goals. The wrapper classes paths corresponding to each environment are defined in gym.envs.int
    if train_env.spec._goal_wrapper_entry_point is not None:
        wrapper_cls = load(train_env.spec._goal_wrapper_entry_point)
        train_env = wrapper_cls(train_env)
        test_env = wrapper_cls(test_env)

    if args['random_seed'] is not None:
        train_env.seed(int(args['random_seed']))
        test_env.seed(int(args['random_seed']))

    # Storing logger output in files with names corresponding to parameters used
    params = args['env'] +'_'+\
        args['memory'] +'_'+\
        args['strategy'] +'_'+\
        args['sampler'] +'_'+\
        args['alpha'] +'_'+\
        args['delta'] +'_'+\
        args['activation'] +'_'+\
        args['invert_grads'] +'_'+\
        args['target_clip'] +'_'+\
        args['sigma']
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Two loggers are defined to retrieve information by step or by episode. Only episodic information is displayed to stdout.
    final_dir = args['summary_dir']+params+'/'+now
    save_dir = args['save_dir']+params+'/'+now
    logger_step = Logger(dir=final_dir+'/log_steps', format_strs=['json'])
    logger_episode = Logger(dir=final_dir+'/log_episodes', format_strs=['stdout', 'json'])

    #TODO integrate the choice of memory in environments specs in gym.env.init
    if args['memory'] == 'sarst':
        memory = SARSTMemory(train_env, limit=int(1e6))
    elif args['memory'] == 'hsarst':
        memory = EpisodicHerSARSTMemory(train_env, limit=int(1e6), strategy=args['strategy'])
    else:
        raise Exception('No existing memory defined')

    # Noise for the actor in vanilla ddpg
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(train_env.action_dim), sigma=float(args['sigma']))

    with tf.Session() as sess:

        actor = ActorNetwork(sess,
                             train_env.state_dim,
                             train_env.action_dim,
                             float(args['tau']),
                             float(args['actor_lr']),
                             args['activation'])

        critic = HuberLossCriticNetwork(sess,
                                        train_env.state_dim,
                                        train_env.action_dim,
                                        float(args['delta']),
                                        float(args['gamma']),
                                        float(args['tau']),
                                        float(args['critic_lr']))

        agent = DDPG_agent(sess,
                           actor,
                           actor_noise,
                           critic,
                           train_env,
                           test_env,
                           memory,
                           args['sampler'],
                           logger_step,
                           logger_episode,
                           int(args['minibatch_size']),
                           int(args['eval_episodes']),
                           int(args['max_steps']),
                           int(args['eval_freq']),
                           save_dir,
                           int(args['save_freq']),
                           int(args['log_freq']),
                           args['target_clip']=='True',
                           args['invert_grads']=='True',
                           float(args['alpha']),
                           args['render_eval'])
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
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default='/home/pierre/PycharmProjects/deep-rl/log/resultsLocal/')
    parser.add_argument('--save-dir', help='directory to store weights of actor and critic',
                        default='/home/pierre/PycharmProjects/deep-rl/log/saveLocal/')
    parser.add_argument('--eval-freq', help='evaluation frequency', default=2000)
    parser.add_argument('--save-freq', help='saving models weights frequency', default=100000)
    parser.add_argument('--log-freq', help='saving models weights frequency', default=2000)
    parser.add_argument('--eval-episodes', help='number of episodes to run during evaluation', default=10)
    parser.add_argument('--eval-steps', help='number of steps in the environment during evaluation', default=200)
    boolean_flag(parser, 'render-eval', default=False)


    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)