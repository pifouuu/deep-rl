import os
import time

import numpy as np
import tensorflow as tf
from .goalSampler2 import RandomGoalSampler, CompetenceProgressGoalBuffer
import pickle


class DDPG_agent():
    def __init__(self,
                 sess,
                 actor,
                 actor_noise,
                 critic, 
                 train_env,
                 test_env,
                 memory,
                 logger_step,
                 logger_episode,
                 logger_memory,
                 batch_size,
                 ep_steps,
                 max_steps,
                 log_dir,
                 save_freq,
                 eval_freq,
                 target_clip,
                 invert_grads,
                 render_test,
                 render_train,
                 train_freq,
                 nb_train_iter,
                 resume_step,
                 resume_timestamp):

        #portrait_actor(actor.target_model, test_env, save_figure=True, figure_file="saved_actor_const.png")
        self.sess = sess
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.target_clip = target_clip
        self.invert_grads = invert_grads

        self.logger_step = logger_step
        self.logger_episode = logger_episode
        self.logger_memory = logger_memory
        self.step_stats = {}
        self.episode_stats = {}
        self.memory_stats = {}
        self.train_freq = train_freq
        self.nb_train_iter = nb_train_iter
        self.ep_steps = ep_steps

        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.log_dir = log_dir

        self.pickle_dir = os.path.join(self.log_dir, 'pickle')
        os.makedirs(self.pickle_dir, exist_ok=True)

        self.resume_step = resume_step
        self.resume_timestamp = resume_timestamp

        self.actor = actor
        self.actor_noise = actor_noise
        self.critic = critic
        self.train_env = train_env
        self.test_env = test_env
        self.memory = memory
        self.render_test = render_test
        self.render_train = render_train

        self.env_step = 0
        self.train_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.nb_goals_reached = 0
        self.video = []
        self.episode_noise_ratio = []

    def train_critic(self, experiences):

        # Calculate targets
        target_q = self.critic.predict_target(
            experiences['state1'],
            self.actor.predict_target(experiences['state1'])
        )

        y_i = []
        for k in range(self.batch_size):
            if self.target_clip:
                target_q[k] = np.clip(target_q[k],
                                      self.train_env.reward_range[0] / (1 - self.critic.gamma),
                                      self.train_env.reward_range[1])

            if experiences['terminal'][k]:
                y_i.append(experiences['reward'][k])
            else:
                y_i.append(experiences['reward'][k] + self.critic.gamma * target_q[k])

        # Update the critic given the targets
        stats = self.critic.train(experiences['state0'], experiences['action'], np.reshape(y_i, (self.batch_size, 1)))
        return target_q, stats

    def train_actor(self, experiences):

        a_outs = self.actor.predict(experiences['state0'])
        q_vals, grads = self.critic.gradients(experiences['state0'], a_outs)
        if self.invert_grads:
            """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
            low = self.train_env.action_space.low
            high = self.train_env.action_space.high
            for d in range(grads[0].shape[0]):
                width = high[d]-low[d]
                for k in range(self.batch_size):
                    if grads[k][d]>=0:
                        grads[k][d] *= (high[d]-a_outs[k][d])/width
                    else:
                        grads[k][d] *= (a_outs[k][d]-low[d])/width
        stats = self.actor.train(experiences['state0'], grads)
        return q_vals, stats

    def update_targets(self):
        self.actor.target_train()
        self.critic.target_train()

    def save(self):
        dir = os.path.join(self.log_dir, 'saves')
        os.makedirs(dir, exist_ok=True)
        self.actor.save_model(os.path.join(dir, 'actor_model.h5'), overwrite=True)
        self.actor.save_target_model(os.path.join(dir, 'target_actor_model.h5'), overwrite=True)
        self.critic.save_model(os.path.join(dir, 'critic_model.h5'), overwrite=True)
        self.critic.save_target_model(os.path.join(dir, 'target_critic_model.h5'), overwrite=True)

    def save_weights(self):
        dir = os.path.join(self.log_dir, 'saves')
        os.makedirs(dir, exist_ok=True)
        self.actor.save_weights(os.path.join(dir, 'actor_weights.h5'), overwrite=True)
        self.actor.save_target_weights(os.path.join(dir, 'target_actor_weights.h5'), overwrite=True)
        self.critic.save_weights(os.path.join(dir, 'critic_weights.h5'), overwrite=True)
        self.critic.save_target_weights(os.path.join(dir, 'target_critic_weights.h5'), overwrite=True)

    def load_weights(self):
        dir = self.log_dir.split('/')[:-1]
        dir.append(self.resume_timestamp)
        dir = '/'.join(dir)
        dir = os.path.join(dir, 'saves')
        self.actor.load_weights(os.path.join(dir, 'actor_weights_{}.h5'.format(self.resume_step)))
        self.critic.load_weights(os.path.join(dir, 'critic_weights_{}.h5'.format(self.resume_step)))
        self.actor.load_target_weights(os.path.join(dir, 'target_actor_weights_{}.h5'.format(self.resume_step)))
        self.critic.load_target_weights(os.path.join(dir, 'target_critic_weights_{}.h5'.format(self.resume_step)))

    def log(self, stats, logger):
        for key in sorted(stats.keys()):
            logger.logkv(key, stats[key])
        logger.dumpkvs()

    def init_var(self):
        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.sess.run(tf.variables_initializer(uninitialized_variables))

    def init_training(self):
        if self.resume_timestamp is not None:
            self.load_weights()
            self.env_step = self.resume_step
            self.episode = 0
            self.nb_goals_reached = 0
        else:
            self.update_targets()

    def reset_train(self):
        if self.train_env.goal_parameterized:
            while True:
                self.train_env.goal = self.memory.sample_goal()
                if self.train_env.is_reachable():
                    break
        state = self.train_env.reset()
        self.start = state[self.train_env.state_to_reached]
        return state

    def run(self):
        self.init_var()
        self.init_training()

        self.start_time = time.time()

        state0 = self.reset_train()

        while self.env_step < self.max_steps:

            if self.render_train:
                self.train_env.render(mode='human')

            action = self.act(state0, train=True)
            state1, reward, terminal, _ = self.train_env.step(action[0])
            experience = self.memory.build_exp(state0, action, state1, reward, terminal)
            self.memory.append(experience)

            self.episode_reward += reward
            self.env_step += 1
            self.episode_step += 1
            state0 = state1

            if (terminal or self.episode_step >= self.ep_steps):

                self.episode += 1
                if terminal:
                    self.nb_goals_reached += 1
                self.memory.end_episode(terminal)
                self.log_episode_stats()


                state0 = self.reset_train()
                self.episode_step = 0
                self.episode_reward = 0
                self.episode_noise_ratio = []

            if self.env_step % self.train_freq == 0 and self.env_step > 3*self.batch_size:

                self.critic_stats, self.actor_stats = self.train()

                if self.env_step % self.eval_freq == 0:

                    if self.test_env.goal_parameterized:
                        self.eval_reward = self.test()
                        # self.eval_reward_1 = self.test('init', 'init')
                        # self.eval_reward_2 = self.test('init', 'uni')
                        # self.eval_reward_3 = self.test('uni', 'init')
                        # self.eval_reward_4 = self.test('uni', 'uni')

                    else:
                        raise RuntimeError

                    self.log_step_stats()
                    self.log_memory_stats()

            if self.env_step % self.save_freq == 0:
                self.save()



    def act(self, state, train=False):
        if train:
            action = self.actor.model.predict(np.reshape(state, (1, self.actor.s_dim[0])))
            val_noise = self.actor_noise()
            self.episode_noise_ratio.append(np.abs(val_noise))
            action += val_noise
        else:
            action = self.actor.target_model.predict(np.reshape(state, (1, self.actor.s_dim[0])))
        action = np.clip(action, self.train_env.action_space.low, self.train_env.action_space.high)
        return action

    def log_step_stats(self):

        critic_stats_mean = self.critic_stats.mean(axis=0)
        actor_stats_mean = self.actor_stats.mean(axis=0)
        for name, stat in zip(self.critic.stat_names, critic_stats_mean):
            self.step_stats[name] = stat
        for name, stat in zip(self.actor.stat_names, actor_stats_mean):
            self.step_stats[name] = stat

        self.step_stats['training_step'] = self.train_step
        self.step_stats['env_step'] = self.env_step
        self.step_stats['reward'] = np.mean(self.eval_reward)


        self.log(self.step_stats, self.logger_step)

    def log_episode_stats(self):
        self.episode_stats['Episode'] = self.episode
        self.episode_stats['Goal'] = self.train_env.goal
        # self.episode_stats['Reachable'] = self.train_env.is_reachable()
        self.episode_stats['Start'] = self.start
        self.episode_stats['Train_reward'] = self.episode_reward
        self.episode_stats['Episode_steps'] = self.episode_step
        self.episode_stats['Goal_reached'] = self.nb_goals_reached
        self.episode_stats['Duration'] = time.time() - self.start_time
        self.episode_stats['Train_step'] = self.train_step
        self.episode_stats['Env_step'] = self.env_step
        self.episode_stats['noise_ratio'] = np.mean(self.episode_noise_ratio)

        self.log(self.episode_stats, self.logger_episode)

    def log_memory_stats(self):
        memory_stats = self.memory.stats()
        memory_stats['episode'] = self.episode
        self.log(memory_stats, self.logger_memory)


    def train(self):
        critic_stats = []
        actor_stats = []
        for _ in range(self.nb_train_iter):
            batch_idxs, experiences = self.memory.sample(self.batch_size)
            target_q_vals, critic_stat = self.train_critic(experiences)
            q_vals, actor_stat = self.train_actor(experiences)
            # if self.train_env.goal_parameterized:
            #     self.memory.update(experiences['state0'], q_vals)
            self.update_targets()
            critic_stats.append(critic_stat)
            actor_stats.append(actor_stat)
        self.train_step += self.nb_train_iter
        return np.array(critic_stats), np.array(actor_stats)



    # def test(self, type=None):
    #
    #     if type == 'random':
    #         self.test_env.set_goal_reachable()
    #     elif type == 'init':
    #         self.test_env.set_goal_init()
    #
    #     state = self.test_env.reset()
    #     ep_test_rewards = []
    #     ep_test_reward = 0
    #
    #     for _ in range(self.nb_test_steps):
    #
    #         if self.render_test:
    #             self.test_env.render(mode='human')
    #
    #         action = self.act(state, train=False)
    #         state, reward, terminal, _ = self.test_env.step(action[0])
    #         ep_test_reward += reward
    #         if (terminal or info['past_limit']):
    #             if type == 'random':
    #                 self.test_env.set_goal_reachable()
    #             elif type == 'init':
    #                 self.test_env.set_goal_init()
    #             state = self.test_env.reset()
    #             ep_test_rewards.append(ep_test_reward)
    #             ep_test_reward = 0
    #
    #     ep_test_rewards.append(ep_test_reward)
    #     if self.test_env.rec is not None: self.test_env.rec.close()
    #     return ep_test_rewards

    def run_test_episode(self):

        self.test_env.goal = self.test_env.sample_test_goal()

        state = self.test_env.reset()
        step = 0
        while True:
            action = self.act(state, train=False)
            state, reward, terminal, _ = self.test_env.step(action[0])
            step += 1
            if step >= self.ep_steps:
                break
            elif terminal:
                return 1
        return 0

    def test(self):
        total_reached = 0
        for episode in range(10):
            reached = self.run_test_episode()
            total_reached += reached
        return total_reached/10







