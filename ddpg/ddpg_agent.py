import tensorflow as tf
import numpy as np

import os
import pickle
import time

# def log(stats):
#     for key in sorted(stats.keys()):
#         logger.record_tabular(key, stats[key])
#     logger.dump_tabular()

EVAL_FREQ = 10
EVAL_EPISODES = 20

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
                 logger_step, 
                 logger_episode, 
                 args):

        self.sess = sess
        self.batch_size = int(args['minibatch_size'])
        self.eval_episodes = int(args['eval_episodes'])
        self.max_episode_steps = int(args['max_episode_steps'])
        self.max_steps = int(args['max_steps'])
        self.with_hindsight = args['with_hindsight']
        self.eval_freq = int(args['eval_freq'])

        self.logger_step = logger_step
        self.logger_episode = logger_episode
        self.step_stats = {}
        self.episode_stats = {}

        self.actor = actor
        self.actor_noise = actor_noise
        self.critic = critic
        self.train_env = train_env
        self.test_env = test_env
        self.env_wrapper = env_wrapper
        self.memory = memory
        self.train_step = 0
        self.episode = 0
        self.episode_step = 0
        self.train_goal = None
        self.test_goal = None
        self.episode_reward = 0
        self.total_reward = 0
        self.goal_reached = 0

    def train_critic(self, samples):

        # Calculate targets
        target_q = self.critic.target_model.predict(
            [samples['state1'], self.actor.target_model.predict(samples['state1'])])

        y_i = []
        for k in range(self.batch_size):
            if samples['terminal1'][k]:
                y_i.append(samples['reward'][k])
            else:
                y_i.append(samples['reward'][k] + self.critic.gamma * target_q[k])

        # Update the critic given the targets
        critic_loss = self.critic.model.train_on_batch(
            [samples['state0'], samples['action']], np.reshape(y_i, (self.batch_size, 1)))
        self.step_stats['Critic loss'] = critic_loss

    def train_actor(self, samples):

        a_outs = self.actor.model.predict(samples['state0'])
        grads = self.critic.gradients(samples['state0'], a_outs)
        self.actor.train(samples['state0'], grads)

    def update_targets(self):
        self.actor.target_train()
        self.critic.target_train()

    def save_weights(self, filepath, overwrite=False):
        print("Saving weights")
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.model.save_weights(actor_filepath, overwrite=overwrite)
        self.critic.model.save_weights(critic_filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        filename, extension = os.path.splitext(filepath)
        actor_filepath = filename + '_actor' + extension
        critic_filepath = filename + '_critic' + extension
        self.actor.model.load_weights(actor_filepath)
        self.critic.model.load_weights(critic_filepath)
        # self.hard_update_target_models()


    def step(self, observation, goal, with_noise, test):
        state0 = self.env_wrapper.process_observation(observation, goal)
        if test:
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
        self.train_critic(samples_train)
        self.train_actor(samples_train)
        self.update_targets()

        # actor_stats = self.actor.get_stats(samples_train)
        # for key in sorted(actor_stats.keys()):
        #     self.step_stats[key] = (actor_stats[key])
        # critic_stats = self.critic.get_stats(samples_train)
        # for key in sorted(critic_stats.keys()):
        #     self.step_stats[key] = (critic_stats[key])

    def test(self):
        test_rewards = []

        for episode in range(self.eval_episodes):

            ep_test_reward = 0
            test_obs = self.test_env.reset()
            self.test_goal = self.env_wrapper.sample_test_goal()

            for k in range(self.max_episode_steps):

                test_obs1, test_sample = self.step(test_obs, self.test_goal, with_noise=False, test=True)

                ep_test_reward += test_sample['reward']

                if test_sample['terminal1']:
                    break
                else:
                    test_obs = test_obs1

            test_rewards.append(ep_test_reward)

        self.step_stats['Test reward'] = np.mean(test_rewards)


    def run(self):

        self.sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        #TODO : soft vs hard update
        self.actor.target_train()
        self.critic.target_train()

        obs0 = self.train_env.reset()
        self.train_goal = self.env_wrapper.sample_goal(obs0)

        while self.train_step < self.max_steps:

            obs1, sample = self.step(obs0, self.train_goal, with_noise=True, test=False)
            # state0 = self.env_wrapper.process_observation(obs0, self.train_goal)
            # action = self.actor.predict(np.reshape(state0, (1, self.actor.s_dim)), with_noise=True)
            # obs1, reward_env, done_env, info = self.train_env.step(action[0])
            # sample = self.env_wrapper.process_step(state0, self.train_goal, action, obs1, reward_env, done_env, info)

            self.memory.append(sample)

            reward = sample['reward']
            self.episode_reward += reward
            self.total_reward += reward

            if self.memory.nb_entries > 3*self.batch_size:

                self.train()

            if self.episode_step >= self.max_episode_steps or sample['terminal1']:

                self.episode_stats['Train reward'] = self.episode_reward
                self.episode_stats['Episode steps'] = self.episode_step

                self.train_env.reset()
                self.train_goal = self.env_wrapper.sample_goal(obs0)

                #TODO :integrate flusing in memory
                if self.with_hindsight:
                    self.memory.flush()

                for key in sorted(self.episode_stats.keys()):
                    self.logger_episode.logkv(key, self.episode_stats[key])
                self.logger_episode.dumpkvs()

                self.episode_step = 0
                self.episode_reward = 0
                self.episode += 1
                if sample['terminal1']: self.goal_reached += 1

            if self.train_step % self.eval_freq == 0:

                self.test()
                print('saving weights')
                self.save_weights(self.logger_step.get_dir()+'_weights.h5', overwrite=True)

            for key in sorted(self.step_stats.keys()):
                self.logger_step.logkv(key, self.step_stats[key])
            self.logger_step.dumpkvs()

            self.train_step += 1
            self.episode_step += 1

            obs0 = obs1




# def train(sess, env, eval_env, args, actor, critic, memory, env_wrapper):
#
#     # Set up summary Ops
#     # summary_ops, summary_vars = build_summaries()
#
#     sess.run(tf.global_variables_initializer())
#     # writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)
#
#     # Initialize target network weights
#     actor.update_target_network()
#     critic.update_target_network()
#
#     # Init stats sample
#     stats_sample = None
#
#     epoch_start_time = time.time()
#     total_env_steps = 0
#     total_train_steps = 0
#     obs = env.reset()
#
#     for step in range(int(args['max_steps'])):
#
#         # Selects a goal for the current episode
#         goal_episode = env_wrapper.sample_goal(obs)
#         #goal_episode = goalSampler.sample(obs)
#         if args['episode_reset']:
#             obs = env.reset()
#         init_state = obs
#
#         ep_reward = 0
#         ep_ave_max_q = []
#         critic_losses = []
#
#
#         for j in range(int(args['max_episode_len'])):
#
#             if args['render_env']:
#                 env.render()
#
#             state0 = env_wrapper.process_observation(obs, goal_episode)
#
#             # Added exploration noise
#             #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
#             action = actor.predict(np.reshape(state0, (1, actor.s_dim)), with_noise=True)
#
#             new_obs, r_env, done_env, info = env.step(action[0])
#             total_env_steps += 1
#             buffer_item = env_wrapper.process_step(state0, goal_episode, action, new_obs, r_env, done_env, info)
#             memory.append(buffer_item)
#
#             obs = new_obs
#             r = buffer_item['reward']
#             ep_reward += r
#
#             # Keep adding experience to the memory until
#             # there are at least minibatch size samples
#             if memory.nb_entries > 3*int(args['minibatch_size']):
#
#                 if stats_sample is None:
#                     # Get a sample and keep that fixed for all further computations.
#                     # This allows us to estimate the change in value for the same set of inputs.
#                     stats_sample = memory.sample(int(args['minibatch_size']))
#
#                 sample = memory.sample(int(args['minibatch_size']))
#
#                 # Calculate targets
#                 target_q = critic.predict_target(
#                     sample['state1'], actor.predict_target(sample['state1']))
#
#                 y_i = []
#                 for k in range(int(args['minibatch_size'])):
#                     if sample['terminal1'][k]:
#                         y_i.append(sample['reward'][k])
#                     else:
#                         y_i.append(sample['reward'][k] + critic.gamma * target_q[k])
#
#                 # Update the critic given the targets
#                 predicted_q_value, critic_loss, _ = critic.train(
#                     sample['state0'], sample['action'], np.reshape(y_i, (int(args['minibatch_size']), 1)))
#                 critic_losses.append(critic_loss)
#
#                 ep_ave_max_q.append(np.amax(predicted_q_value))
#
#                 # Update the actor policy using the sampled gradient
#                 a_outs = actor.predict(sample['state0'])
#                 grads = critic.action_gradients(sample['state0'], a_outs)
#                 actor.train(sample['state0'], grads[0])
#
#                 # Update target networks
#                 actor.update_target_network()
#                 critic.update_target_network()
#
#                 total_train_steps += 1
#
#
#
#             if buffer_item['terminal1']:
#                 obs = env.reset()
#                 break
#
#         if args['with_hindsight']:
#             memory.flush()
#
#         combined_stats = {}
#
#         combined_stats['episode'] = i
#         combined_stats['Env steps'] = total_env_steps
#         combined_stats['Train steps'] = total_train_steps
#
#         if stats_sample is not None:
#             actor_stats = actor.get_stats(stats_sample)
#             for key in sorted(actor_stats.keys()):
#                 combined_stats[key] = (actor_stats[key])
#             critic_stats = critic.get_stats(stats_sample)
#             for key in sorted(critic_stats.keys()):
#                 combined_stats[key] = (critic_stats[key])
#
#         combined_stats['Qmax_value'] = np.mean(ep_ave_max_q)
#         combined_stats['Critic_loss'] = np.mean(critic_losses)
#         combined_stats['Reward'] = ep_reward
#
#
#         print('| Reward: {:d} | Episode: {:d} | Init_state: {:.2f} | Goal: {:.2f} | Duration: {:.4f}'.format(int(ep_reward), \
#                                                                                         i, init_state[0], goal_episode[0],
#                                                                                         time.time() - epoch_start_time))
#
#         if args['eval'] and i % EVAL_FREQ == 0 and i>0:
#
#             eval_rewards = []
#
#             for eval_ep in range(EVAL_EPISODES):
#                 ep_eval_reward = 0
#                 eval_obs = eval_env.reset()
#                 eval_goal = env_wrapper.sample_eval_goal()
#
#                 for k in range(int(args['eval_steps'])):
#
#                     eval_state0 = env_wrapper.process_observation(eval_obs, eval_goal)
#
#                     eval_action = actor.predict_target(np.reshape(eval_state0, (1, actor.s_dim)))
#
#                     new_eval_obs, r_eval_env, done_eval_env, info_eval = eval_env.step(eval_action[0])
#                     eval_buffer_item = env_wrapper.process_step(eval_state0, eval_goal, eval_action, new_eval_obs,
#                                                            r_eval_env, done_eval_env, info_eval)
#
#                     ep_eval_reward += eval_buffer_item['reward']
#
#                     if eval_buffer_item['terminal1']:
#                         break
#                     else:
#                         eval_obs = new_eval_obs
#
#                 eval_rewards.append(ep_eval_reward)
#
#             combined_stats['Eval_reward'] = np.mean(eval_rewards)
#
#         log(combined_stats)