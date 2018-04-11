import tensorflow as tf
import numpy as np
from gym.envs.registration import make
import argparse
import pprint as pp
from ddpg.logger import Logger
from ddpg.memory import SARSTMemory, EpisodicHerSARSTMemory
from ddpg.regionTree3 import FixedRegionsMemory
from ddpg.regionTree3_goal import FixedGoalMemory

import datetime
from ddpg.networks import ActorNetwork, CriticNetwork
from ddpg.ddpgAgent import DDPG_agent
from ddpg.noise import OrnsteinUhlenbeckActionNoise
from gym.spaces import Box
from ddpg.util import load, boolean_flag
import json
import os

def main(args):
    """Despite following the directives of https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development, fully reproducible results could not be obtained. See here : https://github.com/keras-team/keras/issues/2280 for any improvements"""

    params = [str(args['env']),
             str(args['memory']),
             str(args['her']),
             str(args['n_her_goals']),
             str(args['alpha']),
             str(args['n_split']),
             str(args['split_min']),
             str(args['n_window']),
             str(args['sigma']),
             str(args['train_freq']),
             str(args['nb_train_iter']),
             str(args['reward_type']),
             str(args['eps']),
             str(args['sampler']),
             str(args['N']),
             str(args['n_points'])]


    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")

    # Two loggers are defined to retrieve information by step or by episode. Only episodic information is displayed to stdout.
    log_dir = os.path.join(args['log_dir'], '_'.join(params), now)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as config_file:
        config_file.write(json.dumps(args))
    logger_step = Logger(dir=os.path.join(log_dir,'log_steps'), format_strs=['stdout', 'json'])
    logger_episode = Logger(dir=os.path.join(log_dir,'log_episodes'), format_strs=['stdout', 'json'])
    logger_memory = Logger(dir=os.path.join(log_dir,'log_memory'), format_strs=['json'])

    # os.environ['PYTHONHASHSEED'] = '0'
    # if args['random_seed'] is not None:
    #     np.random.seed(int(args['random_seed']))
    #     rn.seed(int(args['random_seed']))
    #     tf.set_random_seed(int(args['random_seed']))

    # Make calls EnvRegistry.make, which builds the environment from its specs defined in gym.envs.init end then builds a timeLimit wrapper around the environment to set the max amount of steps to run
    train_env = make(args['env'])
    test_env = make(args['env'])

    # Wraps each environment in a goal_wrapper to override basic env methods and be able to access goal space properties, or modify the environment simulation according to sampled goals. The wrapper classes paths corresponding to each environment are defined in gym.envs.int
    if train_env.spec._goal_wrapper_entry_point is not None:
        wrapper_cls = load(train_env.spec._goal_wrapper_entry_point)
        train_env = wrapper_cls(train_env, args['reward_type'], float(args['eps']))
        test_env = wrapper_cls(test_env, args['reward_type'], float(args['eps']))

    #TODO integrate the choice of memory in environments specs in gym.env.init
    if args['her'] == 'no':
        memory = SARSTMemory(train_env, limit=int(1e6))
    else:
        memory = EpisodicHerSARSTMemory(train_env, limit=int(1e6), strategy=args['her'],
                                        n_her_goals=int(args['n_her_goals']))

    low = np.concatenate([train_env.observation_space.low, train_env.goal_space.low])
    high = np.concatenate([train_env.observation_space.high, train_env.goal_space.high])
    state_space = Box(low, high)

    # Noise for the actor in vanilla ddpg
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(train_env.action_dim), sigma=float(args['sigma']))

    with tf.Session() as sess:

        if args['random_seed'] is not None:
            np.random.seed(int(args['random_seed']))
            tf.set_random_seed(int(args['random_seed']))
            train_env.seed(int(args['random_seed']))
            test_env.seed(int(args['random_seed']))

        actor = ActorNetwork(sess,
                             train_env.state_dim,
                             train_env.action_dim,
                             float(args['tau']),
                             float(args['actor_lr']))

        critic = CriticNetwork(sess,
                               train_env.state_dim,
                               train_env.action_dim,
                               float(args['gamma']),
                               float(args['tau']),
                               float(args['critic_lr']))

        if args['memory'] == 'fixed_goal':
            memory = FixedGoalMemory(state_space,
                                     train_env.internal,
                                     memory,
                                     actor,
                                     critic,
                                     N=int(args['N']),
                                     n_split=int(args['n_split']),
                                     split_min=float(args['split_min']),
                                     alpha=float(args['alpha']),
                                     maxlen=int(args['n_points']),
                                     n_window=int(args['n_window']),
                                     render=args['render_memory'],
                                     sampler=args['sampler'])

        elif args['memory'] == 'fixed_region':
            memory = FixedRegionsMemory(state_space,
                                        train_env.internal,
                                        memory,
                                        actor,
                                        critic,
                                        N=int(args['N']),
                                        n_split=int(args['n_split']),
                                        split_min=float(args['split_min']),
                                        alpha=float(args['alpha']),
                                        maxlen=int(args['n_points']),
                                        n_window=int(args['n_window']),
                                        render=args['render_memory'],
                                        sampler=args['sampler'])

        else:
            raise RuntimeError

        agent = DDPG_agent(sess,
                           actor,
                           actor_noise,
                           critic,
                           train_env,
                           test_env,
                           memory,
                           logger_step,
                           logger_episode,
                           logger_memory,
                           int(args['minibatch_size']),
                           int(args['ep_steps']),
                           int(args['max_steps']),
                           log_dir,
                           int(args['save_freq']),
                           int(args['eval_freq']),
                           args['target_clip'],
                           args['invert_grads'],
                           args['render_test'],
                           args['render_train'],
                           int(args['train_freq']),
                           int(args['nb_train_iter']),
                           args['resume_step'],
                           args['resume_timestamp'])
        agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # base parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)
    parser.add_argument('--random-seed', help='random seed for repeatability', default=None)
    boolean_flag(parser, 'render-test', default=False)
    boolean_flag(parser, 'render-train', default=False)
    boolean_flag(parser, 'render-memory', default=False)
    boolean_flag(parser, 'invert-grads', default=True)
    boolean_flag(parser, 'target-clip', default=True)

    parser.add_argument('--env', help='choose the gym env', default='CMCPos-v0')
    parser.add_argument('--memory', help='type of memory to use', default='fixed_goal')
    parser.add_argument('--her', help='hindsight strategy: no, final, episode or future', default='no')
    parser.add_argument('--n-her-goals', default=4)
    parser.add_argument('--alpha', default=0)
    parser.add_argument('--n-split', help='number of split comparisons', default=10)
    parser.add_argument('--split-min', help='minimum cp difference to allow split', default=0.0001)
    parser.add_argument('--n-window', help='length of running window used to compute cp', default=5)
    parser.add_argument('--sigma', help="amount of exploration", default=0.3)
    parser.add_argument('--train-freq', help='training frequency', default=1)
    parser.add_argument('--nb-train-iter', help='training iteration number', default=1)
    parser.add_argument('--reward-type', help='sparse, dense', default='sparse')
    parser.add_argument('--eps', default=0.1)
    parser.add_argument('--sampler', help='random, initial, prioritized', default='random')
    parser.add_argument('--N', help='number of regions in goal space', default=16)
    parser.add_argument('--n-points', help='number of points stored in region', default=100)

    parser.add_argument('--max-steps', help='max num of episodes to do while training', default=500000)
    parser.add_argument('--log-dir', help='directory for storing run info',
                        default='/home/pierre/PycharmProjects/deep-rl/log/local/')
    parser.add_argument('--resume-timestamp', help='directory to retrieve weights of actor and critic',
                        default=None)
    parser.add_argument('--resume-step', help='resume_step', default=None)
    parser.add_argument('--ep-steps', help='number of steps in the environment during evaluation', default=50)
    parser.add_argument('--save-freq', help='saving models weights frequency', default=1000)
    parser.add_argument('--eval-freq', help='evaluating every n training steps', default=1000)

    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
