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
from critic import CriticNetwork
from ddpgAgent import DDPG_agent
from noise import OrnsteinUhlenbeckActionNoise
from goalSampler import PrioritizedIntervalBuffer, RandomGoalSampler, NoGoalSampler, InitialGoalSampler, PrioritizedGoalBuffer

#TODO : Update doc on github on this code


def main(args):
    params = 'memory_'+args['memory']+'_goal_'+args['sampler'] +'_wrapper_'+args['wrapper']
    logdir = args['summary_dir']
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    final_dir = logdir+params+'/'+now
    save_dir = args['save_dir']+params+'/'+now
    log_freq = args['log_freq']

    logger_step = Logger(dir=final_dir+'/log_steps', format_strs=['json', 'tensorboard_'+str(log_freq)])
    logger_episode = Logger(dir=final_dir+'/log_episodes', format_strs=['stdout', 'json', 'tensorboard_1'])


    actor_lr = float(args['actor_lr'])
    tau = float(args['tau'])
    critic_lr = float(args['critic_lr'])
    gamma = float(args['gamma'])
    batch_size = int(args['minibatch_size'])
    eval_episodes = int(args['eval_episodes'])
    max_episode_steps = int(args['max_episode_steps'])
    max_steps = int(args['max_steps'])
    eval_freq = int(args['eval_freq'])
    save_freq = int(args['save_freq'])

    train_env = gym.make(args['env'])
    test_env = gym.make(args['env'])

    env_wrapper = None
    if args['wrapper'] == 'NoGoal':
        env_wrapper = NoGoalWrapper()
    elif args['wrapper'] == 'WithGoal':
        env_wrapper = WithGoal()
    elif args['wrapper'] == 'IntervalCurri':
        env_wrapper = IntervalCurriculum()
    elif args['wrapper'] == 'GoalCurri':
        env_wrapper = GoalCurriculum()
    else:
        print("Nooooooooooooooooooooo")

    goal_sampler = None
    if args['sampler'] == 'NoGoal':
        goal_sampler = NoGoalSampler()
    elif args['sampler'] == 'Random':
        goal_sampler = RandomGoalSampler(env_wrapper)
    elif args['sampler'] == 'Initial':
        goal_sampler = InitialGoalSampler(env_wrapper)
    elif args['sampler'] == 'IntervalCurri':
        goal_sampler = PrioritizedIntervalBuffer(int(1e3), 0.5, env_wrapper)
    elif args['sampler'] == 'GoalCurri':
        goal_sampler = PrioritizedGoalBuffer(int(1e3), 0.5, env_wrapper)
    else:
        print("Nooooooo")

    memory = None
    if args['memory'] == 'SAS':
        memory = SASMemory(env_wrapper, limit=int(1e6))
    elif args['memory'] == 'SARST':
        memory = SARSTMemory(env_wrapper, limit=int(1e6))
    elif args['memory'] == 'hindsight_SARST':
        memory = EpisodicHerSARSTMemory(env_wrapper, limit=int(1e6), strategy=args['strategy'])
    elif args['memory'] == 'hindsight_SAS':
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
                             tau,
                             actor_lr)

        # actor.load_weights(args['save_dir']+params+'/2017_11_30_16_56_43/actor_weights_300.h5')

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
                           goal_sampler,
                           logger_step,
                           logger_episode,
                           batch_size,
                           eval_episodes,
                           max_episode_steps,
                           max_steps,
                           eval_freq,
                           save_dir,
                           save_freq,
                           log_freq)
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

    parser.add_argument('--wrapper', help='concatenate goal and observation in states', default='WithGoal')
    parser.add_argument('--memory', help='type of memory to use', default='hindsight_SAS')
    parser.add_argument('--strategy', help='hindsight strategy: final, episode or future', default='future')
    parser.add_argument('--sampler', help='type of goal sampling', default='Random')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=None)
    parser.add_argument('--max-steps', help='max num of episodes to do while training', default=200000)
    parser.add_argument('--max-episode-steps', help='max number of steps before resetting environment', default=200)
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
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
