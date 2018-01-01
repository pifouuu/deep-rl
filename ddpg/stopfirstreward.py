import tensorflow as tf
import numpy as np

import os
import sys
import pickle
import time

from printer import print_status
from plot import portrait_actor

# def log(stats):
#     for key in sorted(stats.keys()):
#         logger.record_tabular(key, stats[key])
#     logger.dump_tabular()

EVAL_FREQ = 10
EVAL_EPISODES = 20

class stop_first_reward():
    def __init__(self,
                 sess,
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
                 eval_freq,
                 save_step_stats,
                 averaging):

        self.sess = sess
        self.batch_size = batch_size
        self.eval_episodes = eval_episodes
        self.max_episode_steps = max_episode_steps
        self.max_steps = max_steps
        self.eval_freq = eval_freq

        self.logger_step = logger_step
        self.logger_episode = logger_episode
        self.step_stats = {}
        self.episode_stats = {}

        self.actor = actor
        self.actor_noise = actor_noise
        self.critic = critic
        self.train_env = train_env
        self.test_env = test_env
        self.save_step_stats = save_step_stats
        self.averaging = averaging

        self.env_wrapper = env_wrapper
        self.memory = memory
        
        self.train_step = 0
        self.episode = 0
        self.episode_step = 0
        self.train_goal = None
        self.test_goal = None
        self.episode_reward = 0
        self.total_reward = 0
        self.nb_goals_reached = 0
        self.episode_init = None


    def train_critic(self, samples):

        # Calculate targets
        target_q = self.critic.predict_target(samples['state1'], self.actor.predict_target(samples['state1']))

        y_i = []
        for k in range(self.batch_size):
            if samples['terminal1'][k]:
                y_i.append(samples['reward'][k])
            else:
                y_i.append(samples['reward'][k] + self.critic.gamma * target_q[k])

        # Update the critic given the targets
        critic_loss = self.critic.train(samples['state0'], samples['action'], np.reshape(y_i, (self.batch_size, 1)))
        self.step_stats['Critic loss'] = critic_loss

        critic_stats = self.critic.get_stats(samples)
        for key in sorted(critic_stats.keys()):
            self.step_stats[key] = (critic_stats[key])

        if (self.step_stats['reference_action_grads'] > 100):
            self.step_stats['divergence'] = self.train_step

    def train_actor(self, samples):
        a_outs = self.actor.predict(samples['state0'])
        grads = self.critic.gradients(samples['state0'], a_outs)
        self.actor.train(samples['state0'], grads)

        actor_stats = self.actor.get_stats(samples)
        for key in sorted(actor_stats.keys()):
            self.step_stats[key] = (actor_stats[key])

    def update_targets(self):
        self.actor.target_train()
        if (self.critic.tau<=1):
            self.critic.target_train()
        elif (self.train_step%self.critic.tau==0):
            self.critic.target_hard_update()

    def save_weights(self, filepath, overwrite=False):
        self.actor.save_weights(filepath, overwrite=overwrite)
        self.critic.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        self.actor.load_weights(filepath)
        self.critic.load_weights(filepath)
        # self.hard_update_target_models()

    def step(self, observation, goal, k=0, test=True):
        state0 = self.env_wrapper.process_observation(observation, goal)
        if test:
            #fig_name = "saved_actor_{}.png".format(k)
            #portrait_actor(self.actor.target_model, self.test_env, save_figure=True, figure_file=fig_name)
            action = self.actor.target_model.predict(np.reshape(state0, (1, self.actor.s_dim)))
            obs1, reward_env, done_env, info = self.test_env.step(action[0])
        else:
            action = self.actor.model.predict(np.reshape(state0, (1, self.actor.s_dim)))
            action += self.actor_noise()
            action = np.clip(action, -self.actor.action_bound, self.actor.action_bound)
            obs1, reward_env, done_env, info = self.train_env.step(action[0])
        sample = self.env_wrapper.process_step(state0, goal, action, obs1, reward_env, done_env, info)
        return obs1, sample

    def train(self):
        samples_train = self.memory.sample(self.batch_size)
        if (self.averaging):
            self.train_critic(samples_train)
        else:
            self.train_critic_no_averaging(samples_train)
        self.train_actor(samples_train)
        self.update_targets()

    def run(self):

        #self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        self.update_targets()

        obs0 = self.train_env.reset()
        self.episode_init = obs0

        self.train_goal = self.env_wrapper.sample_goal()

        print("starting")

        while self.train_step < self.max_steps:

            obs1, sample = self.step(obs0, self.train_goal, test=False)

            self.memory.append(sample)

            reward = sample['reward']
            self.episode_reward += reward
            self.total_reward += reward

            if self.train_step % self.eval_freq == 0:
                print ("step",self.train_step)

            if self.memory.nb_entries > 3*self.batch_size:
                self.train()
            if self.train_step > 300000:
                print("failure!")
                return -1
            if sample['terminal1']:
                print("goal reached:",self.train_step)
                return self.train_step

            elif self.episode_step >= self.max_episode_steps:
                self.train_env.reset()
                self.episode_step = 0
                self.episode_reward = 0
                self.episode += 1

            else:
                obs0 = obs1

            self.train_step += 1
            self.episode_step += 1
            #print_status("{}/{}".format(self.train_step, self.max_steps))



