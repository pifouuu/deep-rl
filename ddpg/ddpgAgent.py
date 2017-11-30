import tensorflow as tf
import numpy as np

import os
import sys
import pickle
import time

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
        self.episode_reward = 0
        self.nb_goals_reached = 0

    def train_critic(self, experiences):

        # Calculate targets
        target_q = self.critic.predict_target(
            experiences['state1'], self.actor.predict_target(experiences['state1']))

        y_i = []
        for k in range(self.batch_size):
            # reward, terminal = self.env_wrapper.eval_exp(experiences['state0'][k],
            #                                              experiences['action'][k],
            #                                              experiences['state1'][k])
            # if terminal:
            #     y_i.append(reward)
            # else:
            #     y_i.append(reward + self.critic.gamma * target_q[k])
            #TODO fix to work with SAS
            if experiences['terminal'][k]:
                y_i.append(experiences['reward'][k])
            else:
                y_i.append(experiences['reward'][k] + self.critic.gamma * target_q[k])

        # Update the critic given the targets
        critic_loss = self.critic.train(
            experiences['state0'], experiences['action'], np.reshape(y_i, (self.batch_size, 1)))
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

    def train(self):
        experiences = self.memory.sample(self.batch_size)

        self.train_critic(experiences)
        self.train_actor(experiences)
        self.update_targets()

        actor_stats = self.actor.get_stats(experiences)
        for key in sorted(actor_stats.keys()):
            self.step_stats[key] = (actor_stats[key])
        critic_stats = self.critic.get_stats(experiences)
        for key in sorted(critic_stats.keys()):
            self.step_stats[key] = (critic_stats[key])

    def run_test_episode(self, test_goal):
        ep_test_reward = 0
        test_obs = self.test_env.reset()
        for k in range(self.max_episode_steps):
            state = self.goal_sampler.process_observation(test_obs, test_goal)
            action = self.actor.target_model.predict(np.reshape(state, (1, self.actor.s_dim)))
            next_obs, reward_env, done_env, info = self.test_env.step(action[0])
            next_state = self.goal_sampler.process_observation(next_obs, test_goal)
            reward, reached = self.env_wrapper.eval_exp(state, action, next_state)
            ep_test_reward += reward
            if reached:
                break
            else:
                test_obs = next_obs
        return ep_test_reward

    def test(self):
        test_rewards = []
        for episode in range(self.eval_episodes):
            test_goal = self.goal_sampler.get_initial_goal()
            reward = self.run_test_episode(test_goal)
            test_rewards.append(reward)
        self.step_stats['Test reward on initial goal'] = np.mean(test_rewards)

        if self.env_wrapper.has_goal:
            test_rewards = []
            for episode in range(self.eval_episodes):
                test_goal = self.goal_sampler.get_random_goal()
                reward = self.run_test_episode(test_goal)
                test_rewards.append(reward)
            self.step_stats['Test reward on random goal'] = np.mean(test_rewards)

    def run(self):

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        #TODO : soft vs hard update
        self.actor.target_train()
        self.critic.target_train()

        #TODO : load actor and critic if need be

        current_obs = self.train_env.reset()
        episode_init = current_obs
        train_goal = self.goal_sampler.sample()
        start_time = time.time()

        while self.train_step < self.max_steps:
            state = self.goal_sampler.process_observation(current_obs, train_goal)
            action = self.actor.model.predict(np.reshape(state, (1, self.actor.s_dim)))
            action += self.actor_noise()
            action = np.clip(action, -self.actor.action_bound, self.actor.action_bound)
            next_obs, reward_env, done_env, info = self.train_env.step(action[0])
            next_state = self.goal_sampler.process_observation(next_obs, train_goal)
            reward, reached = self.env_wrapper.eval_exp(state, action, next_state)

            experience = self.memory.build_exp(state, action, next_state, reward, reached)

            self.memory.append(experience)
            self.episode_reward += reward
            current_obs = next_obs
            self.train_step += 1
            self.episode_step += 1

            if self.memory.nb_entries > 3*self.batch_size:
                self.train()

            if self.episode_step >= self.max_episode_steps or reached:

                self.episode += 1
                if reached: self.nb_goals_reached += 1
                self.episode_stats['Episode'] = self.episode
                self.episode_stats['Start'] = episode_init[0]
                self.episode_stats['Goal'] = train_goal[0]
                self.episode_stats['Train reward'] = self.episode_reward
                self.episode_stats['Episode steps'] = self.episode_step
                self.episode_stats['Goal reached'] = self.nb_goals_reached
                self.episode_stats['Duration'] = time.time() - start_time

                current_obs = self.train_env.reset()
                episode_init = current_obs
                train_goal = self.goal_sampler.sample()
                self.memory.end_episode()

                for key in sorted(self.episode_stats.keys()):
                    self.logger_episode.logkv(key, self.episode_stats[key])
                self.logger_episode.dumpkvs()

                self.episode_step = 0
                self.episode_reward = 0

            if self.train_step % self.eval_freq == 0:
                self.test()

            #TODO less train stats
            self.step_stats['Training steps'] = self.train_step
            for key in sorted(self.step_stats.keys()):
                self.logger_step.logkv(key, self.step_stats[key])
            self.logger_step.dumpkvs()



