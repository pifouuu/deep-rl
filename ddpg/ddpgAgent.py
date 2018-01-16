import os
import time

import numpy as np
import tensorflow as tf
from ddpg.goalSampler2 import RandomGoalSampler, \
    PrioritizedGoalBuffer, CompetenceProgressGoalBuffer


def gradient_inverter(gradient, p_min, p_max):
    """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
    delta = p_max - p_min
    if delta <= 0:
        raise(ValueError("p_max <= p_min"))

    inverted_gradient = tf.where(gradient >= 0, (p_max - gradient) / delta, (gradient - p_min) / delta)

    return(inverted_gradient)

class DDPG_agent():
    def __init__(self,
                 sess,
                 actor,
                 actor_noise,
                 critic, 
                 train_env,
                 test_env,
                 memory,
                 goal_sampler,
                 logger_step,
                 logger_episode,
                 batch_size,
                 eval_episodes,
                 max_steps,
                 eval_freq,
                 save_dir,
                 save_freq,
                 log_freq,
                 target_clip,
                 invert_grads,
                 alpha):

        #portrait_actor(actor.target_model, test_env, save_figure=True, figure_file="saved_actor_const.png")
        self.sess = sess
        self.batch_size = batch_size
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.eval_freq = eval_freq
        self.target_clip = target_clip
        self.invert_grads = invert_grads

        self.logger_step = logger_step
        self.logger_episode = logger_episode
        self.step_stats = {}
        self.episode_stats = {}
        self.log_freq = log_freq

        self.save_freq = save_freq
        self.save_dir = save_dir

        self.actor = actor
        self.actor_noise = actor_noise
        self.critic = critic
        self.train_env = train_env
        self.test_env = test_env

        #portrait_actor(self.actor.target_model, self.test_env, save_figure=True, figure_file="saved_actor_const2.png")
        self.memory = memory
        self.alpha = alpha

        if goal_sampler == 'no' or goal_sampler == 'rnd':
            self.goal_sampler = RandomGoalSampler(self.train_env)
        elif goal_sampler == 'goalC':
            self.goal_sampler = PrioritizedGoalBuffer(int(1e3), self.alpha, self.train_env)
        elif goal_sampler == 'comp':
            self.goal_sampler = CompetenceProgressGoalBuffer(int(1e3), self.alpha,
                                                             self.train_env,
                                                             self.actor,
                                                             self.critic)

        self.train_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.nb_goals_reached = 0

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
        critic_loss = self.critic.train(
            experiences['state0'], experiences['action'], np.reshape(y_i, (self.batch_size, 1)))
        self.step_stats['list/critic_loss'].append(critic_loss)

        return target_q

    def train_actor(self, samples):

        a_outs = self.actor.predict(samples['state0'])
        q_vals, grads = self.critic.gradients(samples['state0'], a_outs)
        if self.invert_grads:
            low = self.train_env.action_space.low
            high = self.train_env.action_space.high
            for d in range(grads[0].shape[0]):
                width = high[d]-low[d]
                for k in range(self.batch_size):
                    if grads[k][d]>=0:
                        grads[k][d] *= (high[d]-a_outs[k][d])/width
                    else:
                        grads[k][d] *= (a_outs[k][d]-low[d])/width
        self.actor.train(samples['state0'], grads)

        return q_vals

    def update_targets(self):
        self.actor.target_train()
        self.critic.target_train()


    def load_weights(self, filepath):
        self.actor.load_weights(filepath)
        self.critic.load_weights(filepath)
        # self.hard_update_target_models()

    def train(self):
        batch_idxs, experiences = self.memory.sample(self.batch_size)

        target_q_vals = self.train_critic(experiences)
        q_vals = self.train_actor(experiences)

        self.update_targets()

        actor_stats = self.actor.get_stats(experiences)
        for key in sorted(actor_stats.keys()):
            self.step_stats['list/'+key].append(actor_stats[key])
        critic_stats = self.critic.get_stats(experiences)
        for key in sorted(critic_stats.keys()):
            self.step_stats['list/'+key].append(critic_stats[key])

    def run_test_episode(self, goal):
        ep_test_reward = 0
        state_0 = self.test_env.reset()

        while True:

            agent_state_0 = np.concatenate([state_0, goal])

            action = self.actor.model.predict(np.reshape(agent_state_0, (1, self.actor.s_dim)))
            action = np.clip(action, self.test_env.action_space.low, self.test_env.action_space.high)

            state_1, reward, terminal, info = self.test_env.step(action[0])
            agent_state_1 = np.concatenate([state_1, goal])

            agent_reward, agent_terminal = self.test_env.eval_exp(agent_state_0, action, agent_state_1, reward,
                                                                   terminal)
            reward = agent_reward
            terminal = agent_terminal or (terminal and info['past_limit'])
            ep_test_reward += reward
            if terminal:
                break
            else:
                state_0 = state_1
        return ep_test_reward

    def test(self):
        test_rewards = []
        for episode in range(self.eval_episodes):
            test_goal = self.test_env.get_initial_goal()
            reward = self.run_test_episode(test_goal)
            test_rewards.append(reward)
        self.step_stats['Test reward on initial goal'] = np.mean(test_rewards)

        if len(self.test_env.state_to_goal) > 0:
            test_rewards = []
            for episode in range(self.eval_episodes):
                test_goal = self.test_env.get_random_goal()
                reward = self.run_test_episode(test_goal)
                test_rewards.append(reward)
            self.step_stats['Test reward on random goal'] = np.mean(test_rewards)

    def save_models(self):
        dir = self.save_dir+'/'
        os.makedirs(dir, exist_ok=True)

        self.actor.save_model(dir+'actor_model_{}.h5'.format(self.train_step), overwrite=True)
        self.actor.save_target_model(dir + 'target_actor_model_{}.h5'.format(self.train_step), overwrite=True)
        self.critic.save_model(dir + 'critic_model_{}.h5'.format(self.train_step), overwrite=True)
        self.critic.save_target_model(dir + 'target_critic_model_{}.h5'.format(self.train_step), overwrite=True)

    def save_weights(self):
        dir = self.save_dir+'/'
        os.makedirs(dir, exist_ok=True)
        self.actor.save_weights(dir+'actor_weights_{}.h5'.format(self.train_step), overwrite=True)
        self.actor.save_target_weights(dir + 'target_actor_weights_{}.h5'.format(self.train_step), overwrite=True)
        self.critic.save_weights(dir + 'critic_weights_{}.h5'.format(self.train_step), overwrite=True)
        self.critic.save_target_weights(dir + 'target_critic_weights_{}.h5'.format(self.train_step), overwrite=True)

    def save_archi(self):
        dir = self.save_dir+'/'
        os.makedirs(dir, exist_ok=True)

        self.actor.save_archi(dir+'actor_archi.json', overwrite=True)
        self.actor.save_target_archi(dir + 'target_actor_archi.json', overwrite=True)
        self.critic.save_archi(dir + 'critic_archi.json', overwrite=True)
        self.critic.save_target_archi(dir + 'target_critic_archi.json', overwrite=True)

    def run(self):

        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.sess.run(tf.variables_initializer(uninitialized_variables))

        self.step_stats['list/critic_loss'] = []
        for stat in self.actor.stat_names:
            self.step_stats['list/'+stat] = []
        for stat in self.critic.stat_names:
            self.step_stats['list/'+stat] = []

        # Initialize target network weights
        #TODO : soft vs hard update
        self.actor.target_train()
        self.critic.target_train()

        state_0 = self.train_env.reset()
        goal = self.goal_sampler.sample()
        agent_state_0 = np.concatenate([state_0, goal])
        episode_init = agent_state_0

        self.save_archi()
        start_time = time.time()

        while self.train_step < self.max_steps:

            action = self.actor.model.predict(np.reshape(agent_state_0, (1, self.actor.s_dim)))
            action += self.actor_noise()
            action = np.clip(action, self.train_env.action_space.low, self.train_env.action_space.high)

            state_1, reward, terminal, info = self.train_env.step(action[0])
            agent_state_1 = np.concatenate([state_1, goal])
            agent_reward, agent_terminal = self.train_env.eval_exp(agent_state_0, action, agent_state_1, reward, terminal)
            reward = agent_reward
            terminal = agent_terminal or (terminal and info['past_limit'])

            experience = self.memory.build_exp(agent_state_0, action, agent_state_1, agent_reward, agent_terminal)

            self.memory.append(experience)
            self.episode_reward += reward
            agent_state_0 = np.concatenate([state_1,goal])
            self.train_step += 1
            self.episode_step += 1

            if self.memory.nb_entries > 3*self.batch_size:
                self.train()

            if terminal:

                self.episode += 1
                if not info['past_limit']:
                    self.nb_goals_reached += 1
                self.episode_stats['Episode'] = self.episode
                self.episode_stats['Start'] = episode_init[self.train_env.state_to_obs]
                self.episode_stats['Goal'] = episode_init[self.train_env.state_to_goal]
                self.episode_stats['Train reward'] = self.episode_reward
                self.episode_stats['Episode steps'] = self.episode_step
                self.episode_stats['Goal reached'] = self.nb_goals_reached
                self.episode_stats['Duration'] = time.time() - start_time
                self.episode_stats['Train step'] = self.train_step

                state_0 = self.train_env.reset()
                goal = self.goal_sampler.sample()
                agent_state_0 = np.concatenate([state_0, goal])
                episode_init = agent_state_0
                self.memory.end_episode(not info['past_limit'])

                for key in sorted(self.episode_stats.keys()):
                    self.logger_episode.logkv(key, self.episode_stats[key])
                self.logger_episode.dumpkvs()

                self.episode_step = 0
                self.episode_reward = 0

            if self.train_step % self.eval_freq == 0:
                self.test()

            if self.train_step % self.save_freq == 0:
                self.save_models()
                self.save_weights()

            if self.train_step % self.log_freq == 0:
                self.step_stats['training_step'] = self.train_step
                for key, val in self.goal_sampler.stats.items():
                    self.step_stats[key] = val
                for key in sorted(self.step_stats.keys()):
                    if key.startswith('list'):
                        log_key = key.split('/')[1]
                        self.logger_step.logkv(log_key, np.mean(self.step_stats[key]))
                        self.step_stats[key] = []
                    else:
                        self.logger_step.logkv(key, self.step_stats[key])
                self.logger_step.dumpkvs()





