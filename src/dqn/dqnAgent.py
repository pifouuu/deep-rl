import os
import time

import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt


class DQN_Agent():
    def __init__(self,
                 sess,
                 critic,
                 train_env,
                 test_env,
                 memory,
                 logger_step,
                 logger_episode,
                 batch_size,
                 nb_test_steps,
                 max_steps,
                 log_dir,
                 save_freq,
                 eval_freq,
                 target_clip,
                 alpha,
                 render_test,
                 train_freq,
                 nb_train_iter,
                 resume_step,
                 resume_timestamp,
                 start_epsilon,
                 end_epsilon):

        self.sess = sess
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.target_clip = target_clip

        self.logger_step = logger_step
        self.logger_episode = logger_episode
        self.step_stats = {}
        self.episode_stats = {}
        self.train_freq = train_freq
        self.nb_train_iter = nb_train_iter
        self.nb_test_steps = nb_test_steps

        self.save_freq = save_freq
        self.eval_freq = eval_freq

        self.log_dir = log_dir
        self.resume_step = resume_step
        self.resume_timestamp = resume_timestamp

        self.critic = critic
        self.train_env = train_env
        self.test_env = test_env
        self.memory = memory
        self.alpha = alpha
        self.render_test = render_test
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon = 1

        self.train_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.nb_goals_reached = 0

    def train_critic(self, experiences):

        # Calculate targets
        target_action_vals = self.critic.predict_target_action_values(experiences['state1'])
        next_actions = self.critic.select_actions(experiences['state1'])

        y_i = []
        for k in range(self.batch_size):
            target_q = target_action_vals[k][next_actions[k]]
            # if self.target_clip:
            #     target_q[k] = np.clip(target_q[k],
            #                           self.train_env.reward_range[0] / (1 - self.critic.gamma),
            #                           self.train_env.reward_range[1])

            if experiences['terminal'][k]:
                y_i.append(experiences['reward'][k])
            else:
                y_i.append(experiences['reward'][k] + self.critic.gamma * target_q)

        # Update the critic given the targets
        self.critic.train(experiences['state0'], experiences['action'], np.reshape(y_i, (self.batch_size, 1)))

    def update_targets(self):
        self.critic.target_train()

    def save_weights(self):
        dir = self.log_dir+'/saves'
        os.makedirs(dir, exist_ok=True)
        self.critic.save_weights(dir + '/critic_weights_{}.h5'.format(self.train_step), overwrite=True)
        self.critic.save_target_weights(dir + '/target_critic_weights_{}.h5'.format(self.train_step), overwrite=True)

    def load_weights(self):
        dir = self.log_dir.split('/')[:-1]
        dir.append(self.resume_timestamp)
        dir = '/'.join(dir)
        dir = dir+'/saves'
        self.critic.load_weights(dir + '/critic_weights_{}.h5'.format(self.resume_step))
        self.critic.load_target_weights(dir + '/target_critic_weights_{}.h5'.format(self.resume_step))

    def log(self, stats, logger):
        for key in sorted(stats.keys()):
            logger.logkv(key, stats[key])
        logger.dumpkvs()

    def run(self):

        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.sess.run(tf.variables_initializer(uninitialized_variables))

        if self.resume_timestamp is not None:
            self.load_weights()
            self.train_step = self.resume_step
            self.episode = 0
            self.nb_goals_reached = 0
        else:
            self.update_targets()

        state = self.train_env.reset()
        prev_state = state
        self.start_time = time.time()
        while self.train_step < self.max_steps:
            action = self.act(state, noise=True)
            state, reward, terminal = self.train_env.step(action)
            self.episode_reward += reward
            self.train_step += 1
            self.episode_step += 1
            experience = self.memory.build_exp(prev_state, action, state, reward, terminal)
            past_limit = self.episode_step >= 50
            terminal = terminal or past_limit
            self.memory.append(experience)
            prev_state = state

            if terminal:
                self.episode += 1
                if not past_limit:
                    self.nb_goals_reached += 1

                self.log_episode_stats()

                state = self.train_env.reset()

                self.memory.end_episode(not past_limit)

                self.episode_step = 0
                self.episode_reward = 0

            if self.train_step > 10000:

                if self.epsilon > self.end_epsilon:
                    self.epsilon -= (self.start_epsilon - self.end_epsilon)/10000

                if self.train_step % self.train_freq == 0:
                    self.train()
                    # self.eval_rewards_random = self.test()
                    # self.log_step_stats()

                if self.train_step % self.save_freq == 0:
                    self.save_weights()

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.train_env.actions)
        else:
            action = self.critic.select_actions(np.reshape(state, (-1,)+self.train_env.state_dim))[0]
        return action

    # def log_step_stats(self):
    #     self.step_stats['training_step'] = self.train_step
    #     self.step_stats['Test reward on random goal'] = np.mean(self.eval_rewards_random)
    #     self.log(self.step_stats, self.logger_step)

    def log_episode_stats(self):
        self.episode_stats['Episode'] = self.episode
        self.episode_stats['Train reward'] = self.episode_reward
        self.episode_stats['Episode steps'] = self.episode_step
        self.episode_stats['Goal reached'] = self.nb_goals_reached
        self.episode_stats['Duration'] = time.time() - self.start_time
        self.episode_stats['Train step'] = self.train_step
        self.log(self.episode_stats, self.logger_episode)

    def train(self):
        for _ in range(self.nb_train_iter):
            batch_idxs, experiences = self.memory.sample(self.batch_size)
            self.train_critic(experiences)
            self.update_targets()

    def test(self):

        video = np.zeros((self.nb_test_steps,)+self.test_env.state_dim, dtype=np.uint8)

        state = self.test_env.reset()
        ep_test_rewards = []
        ep_test_reward = 0
        step = 0
        for i in range(self.nb_test_steps):
            action = self.act(state, noise=False)
            state, reward, terminal = self.test_env.step(action)
            step += 1
            past_limit = step >= 50
            terminal = terminal or past_limit
            video[i] = state
            ep_test_reward += reward
            if terminal:
                state = self.test_env.reset()
                step = 0
                ep_test_rewards.append(ep_test_reward)
                ep_test_reward = 0

        # if self.render_test:
        #     tic = time.time()
        #     for i in range(self.nb_test_steps):
        #         if i == 0:
        #             img = plt.imshow(video[i])
        #         else:
        #             img.set_data(video[i])
        #         toc = time.time()
        #         clock_dt = toc - tic
        #         tic = time.time()
        #         # Real-time playback not always possible as clock_dt > .03
        #         plt.pause(max(0.01, 0.03 - clock_dt))  # Need min display time > 0.0.
        #         plt.draw()
        #     plt.waitforbuttonpress()

        return ep_test_rewards