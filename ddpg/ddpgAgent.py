import tensorflow as tf
import numpy as np

import os
import sys
import pickle
import time

from printer import print_status
from plot import portrait_actor

class DDPG_agent():
    def __init__(self,
                 sess,
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
                 eval_freq):

        #portrait_actor(actor.target_model, test_env, save_figure=True, figure_file="saved_actor_const.png")
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
        self.goal_sampler = goal_sampler

        #portrait_actor(self.actor.target_model, self.test_env, save_figure=True, figure_file="saved_actor_const2.png")
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
        self.current_obs = None

    def train_critic(self, samples):

        # Calculate targets
        target_q = self.critic.predict_target(
            samples['state1'], self.actor.predict_target(samples['state1']))

        y_i = []
        for k in range(self.batch_size):
            if samples['terminal1'][k]:
                y_i.append(samples['reward'][k])
            else:
                y_i.append(samples['reward'][k] + self.critic.gamma * target_q[k])

        # Update the critic given the targets
        critic_loss = self.critic.train(
            samples['state0'], samples['action'], np.reshape(y_i, (self.batch_size, 1)))
        self.step_stats['Critic loss'] = critic_loss

    def train_actor(self, samples):

        a_outs = self.actor.predict(samples['state0'])
        # TODO : experiment with inverted gradients
        grads = self.critic.gradients(samples['state0'], a_outs)
        self.actor.train(samples['state0'], grads)

    def update_targets(self):
        self.actor.target_train()
        self.critic.target_train()

    #TODO: save 4 networks in one shot with different filenames
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
        self.current_obs = obs1
        return sample

    def train(self):
        samples_train = self.memory.sample(self.batch_size)
        samples_train['rewards'], samples_train['terminals1'] = \
            self.env_wrapper.evaluate_transition(samples_train['state0'],
                                                 samples_train['action'],
                                                 samples_train['state1'])

        self.train_critic(samples_train)
        self.train_actor(samples_train)
        self.update_targets()

        actor_stats = self.actor.get_stats(samples_train)
        for key in sorted(actor_stats.keys()):
            self.step_stats[key] = (actor_stats[key])
        critic_stats = self.critic.get_stats(samples_train)
        for key in sorted(critic_stats.keys()):
            self.step_stats[key] = (critic_stats[key])

    def test(self):
        test_rewards = []

        for episode in range(self.eval_episodes):

            ep_test_reward = 0
            test_obs = self.test_env.reset()
            self.test_goal = self.env_wrapper.sample_initial_goal()
            #fig_name = "saved_actor_ante.png"
            #portrait_actor(self.actor.target_model, self.test_env, save_figure=True, figure_file=fig_name)

            for k in range(self.max_episode_steps):
                test_obs1, test_sample = self.step(test_obs, self.test_goal, k, test=True)
                ep_test_reward += test_sample['reward']
                if test_sample['terminal1']:
                    break
                else:
                    test_obs = test_obs1

            test_rewards.append(ep_test_reward)

        self.step_stats['Test reward on initial goal'] = np.mean(test_rewards)

        test_rewards = []

        for episode in range(self.eval_episodes):

            ep_test_reward = 0
            test_obs = self.test_env.reset()
            self.test_goal = self.env_wrapper.sample_random_goal(test_obs)

            for k in range(self.max_episode_steps):
                test_obs1, test_sample = self.step(test_obs, self.test_goal, test=True)
                ep_test_reward += test_sample['reward']
                if test_sample['terminal1']:
                    break
                else:
                    test_obs = test_obs1

            test_rewards.append(ep_test_reward)

        self.step_stats['Test reward on random goal'] = np.mean(test_rewards)


    def run(self):

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        #TODO : soft vs hard update
        self.actor.target_train()
        self.critic.target_train()

        #TODO : load actor and critic if need be

        self.current_obs = self.train_env.reset()
        self.episode_init = self.current_obs

        self.train_goal = self.goal_sampler.sample()

        while self.train_step < self.max_steps:
            state = self.env_wrapper.process_observation(self.current_obs, self.train_goal)
            action = self.actor.model.predict(np.reshape(state, (1, self.actor.s_dim)))
            action += self.actor_noise()
            action = np.clip(action, -self.actor.action_bound, self.actor.action_bound)
            next_obs, reward_env, done_env, info = self.train_env.step(action[0])
            sample = self.env_wrapper.process_step(state, self.train_goal, action, next_obs, reward_env, done_env, info)

            self.memory.append(sample)

            reward = sample['reward']
            self.episode_reward += reward
            self.total_reward += reward

            if self.memory.nb_entries > 3*self.batch_size:
                self.train()

            if self.episode_step >= self.max_episode_steps or sample['terminal1']:

                self.episode_stats['Episode'] = self.episode
                self.episode_stats['Start'] = self.episode_init[0]
                self.episode_stats['Goal'] = self.train_goal[0]
                self.episode_stats['Train reward'] = self.episode_reward
                self.episode_stats['Episode steps'] = self.episode_step
                #self.episode_stats['Episode difficulty'] = difficulty
                self.episode_stats['Goal reached'] = self.goal_reached

                self.current_obs = self.train_env.reset()
                self.train_goal = self.env_wrapper.sample_goal(self.current_obs, self.goal_reached)

                #TODO :integrate flusing in memory
                self.memory.end_episode()

                for key in sorted(self.episode_stats.keys()):
                    self.logger_episode.logkv(key, self.episode_stats[key])
                self.logger_episode.dumpkvs()

                self.episode_step = 0
                self.episode_reward = 0
                self.episode += 1
                if sample['terminal1']: self.goal_reached += 1

            if self.train_step % self.eval_freq == 0:
                self.test()


            self.step_stats['Training steps'] = self.train_step
            for key in sorted(self.step_stats.keys()):
                self.logger_step.logkv(key, self.step_stats[key])
            self.logger_step.dumpkvs()

            self.train_step += 1
            self.episode_step += 1
