import tensorflow as tf
import numpy as np
import gym
import argparse
import pprint as pp
from logger import Logger
from memory import Memory
import pickle
from envWrapper import NoGoal
from perf_config_mcc import PerfConfig
from actor import ActorNetwork
from HLcriticNetwork import HuberLossCriticNetwork
from myddpgAgent import DDPG_agent
from noise import OrnsteinUhlenbeckActionNoise, NoNoise

from rl.utils.printer import print_info
import os

# Configuration
config = PerfConfig()


def perf_study(delta_clip, num):
    # Get the environment and extract the number of actions.
    env = config.env
    if not config.random_seed:
        np.random.seed(123)
        env.seed(123)

    results_path = './experiments/{}/{}/'.format(delta_clip, num)
    #logger_step = Logger(dir=results_path,format_strs=['log','json', 'tensorboard'])
    #logger_episode = Logger(dir=results_path, format_strs=['log','stdout', 'json', 'tensorboard'])
    logger_step = Logger(dir=results_path,format_strs=['json'])
    logger_episode = Logger(dir=results_path, format_strs=['json'])
        
    # 
    env_wrapper = NoGoal()
    
    state_dim = env_wrapper.state_shape[0]
    action_dim = env_wrapper.action_shape[0]
    action_bound = env.action_space.high
    # Ensure action bound is symmetric
    assert (env.action_space.high == -env.action_space.low)

    memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))
    memory.load_from_ManceronBuffer(file=config.memory_file)

    # Noise
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    #actor_noise = NoNoise()

    with tf.Session() as sess:

        if not config.random_seed:
            np.random.seed(config.seed)
            tf.set_random_seed(config.seed)
            env.seed(config.seed)

        actor = ActorNetwork(sess,
                             state_dim,
                             action_dim,
                             action_bound,
                             config.tau,
                             config.actor_lr)

        critic = HuberLossCriticNetwork(delta_clip,
                                        sess,
                                        state_dim,
                                        action_dim,
                                        config.gamma,
                                        config.tau,
                                        config.critic_lr)

        agent = DDPG_agent(sess,
                           actor,
                           actor_noise,
                           critic,
                           env,
                           env,
                           env_wrapper,
                           memory,
                           logger_step,
                           logger_episode,
                           config.batch_size,
                           config.eval_episodes,
                           config.max_episode_steps,
                           config.max_steps,
                           config.eval_freq)

        agent.run()
    
