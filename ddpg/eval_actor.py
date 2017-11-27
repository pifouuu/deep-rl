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
from critic import CriticNetwork
from myddpgAgent import DDPG_agent
from noise import OrnsteinUhlenbeckActionNoise, NoNoise

from rl.utils.printer import print_info
import os

# Configuration
config = PerfConfig()
env = config.env

results_path = './eval/'
logger_step = Logger(dir=results_path,format_strs=['json'])
logger_episode = Logger(dir=results_path, format_strs=[])
                
env_wrapper = NoGoal()

state_dim = env_wrapper.state_shape[0]
action_dim = env_wrapper.action_shape[0]
action_bound = env.action_space.high
# Ensure action bound is symmetric
assert (env.action_space.high == -env.action_space.low)

memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))

# Noise
#actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
actor_noise = NoNoise()

class Evaluator:

        def eval(self,filename):
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

                        print("evaluating : ", filename)
                        #actor.print_target_weights()
                        actor.load_target_weights(filename)
                        #actor.print_target_weights()
                        actor.load_weights(filename)

                        critic = CriticNetwork(sess,
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
                        for i in range(5):
                                agent.test()


path = "./actors/"
collec = Evaluator()
for actor in os.listdir(path):
    collec.eval(path+actor)
