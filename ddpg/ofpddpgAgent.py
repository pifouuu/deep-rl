from myddpgAgent import DDPG_agent
from plot import portrait_actor

import tensorflow as tf
import numpy as np

# def log(stats):
#     for key in sorted(stats.keys()):
#         logger.record_tabular(key, stats[key])
#     logger.dump_tabular()

EVAL_FREQ = 10
EVAL_EPISODES = 20


class OFPDDPG_agent(DDPG_agent):
    def __init__(self, update_actor, sess, actor, actor_noise, critic, env, env_wrapper, memory, logger_step,
                 logger_episode, batch_size, eval_episodes, max_episode_steps, max_steps, eval_freq, save_step_stats):
        super(OFPDDPG_agent, self).__init__(sess, actor, actor_noise, critic, env, env, env_wrapper, memory,
                                            logger_step, logger_episode, batch_size, eval_episodes, max_episode_steps,
                                            max_steps, eval_freq, save_step_stats)
        self.update_actor = update_actor

    def train_critic(self, samples):

        # Calculate targets
        target_q = self.critic.predict_target(samples['state1'], self.update_actor.predict_target(samples['state1']))

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

    def test(self):
        test_rewards = []
        #print ("in test :", self.memory.size())

        for episode in range(self.eval_episodes):

            ep_test_reward = 0
            test_obs = self.test_env.reset()
            self.test_goal = self.env_wrapper.sample_initial_goal()
            # fig_name = "saved_actor_ante.png"
            # portrait_actor(self.actor.target_model, self.test_env, save_figure=True, figure_file=fig_name)
            max_eval_steps = int(self.max_episode_steps / 5)

            for k in range(max_eval_steps):
                test_obs1, test_sample = self.step(test_obs, self.test_goal, k, test=True)
                ep_test_reward += test_sample['reward']
                if test_sample['terminal1']:
                    break
                else:
                    test_obs = test_obs1
                    # print_status("{}/{}".format(k, self.max_episode_steps))
            test_rewards.append(ep_test_reward)

        mean_reward = np.mean(test_rewards)
        self.episode_stats['New Training steps'] = self.train_step
        self.episode_stats['New Test reward'] = mean_reward
        self.step_stats['Test reward'] = mean_reward
        #print ('Test reward', mean_reward)
        if mean_reward > 97.5:
            self.actor.save_target_weights("actors/good_actor_{}.save".format(mean_reward), overwrite=True)
                # portrait_actor(self.actor.target_model,self.test_env,save_figure=True)
                # self.actor.print_target_weights()
                # sys.exit()

        for key in sorted(self.episode_stats.keys()):
            self.logger_episode.logkv(key, self.episode_stats[key])
        self.logger_episode.dumpkvs()

    def run(self):

        #self.sess.run(tf.global_variables_initializer())
        #portrait_actor(self.update_actor.target_model, self.test_env, save_figure=True, figure_file="./img/update_actor2.png")
        # Initialize target network weights
        #TODO : soft vs hard update
        self.actor.target_train()
        self.critic.target_train()

        #TODO : load actor and critic if need be

        obs0 = self.train_env.reset()
        self.episode_init = obs0

        #TODO : pass on to a sample goal function in the agent, not in the wrapper
        self.train_goal = self.env_wrapper.sample_goal()

        #print ("in run :", self.memory.size())
        while self.train_step < self.max_steps:

            obs1, sample = self.step(obs0, self.train_goal, test=False)

            reward = sample['reward']
            self.episode_reward += reward
            self.total_reward += reward

            if self.memory.nb_entries > 3*self.batch_size:
                self.train()

            if self.train_step % self.eval_freq == 0:
                self.test()

            if self.episode_step >= self.max_episode_steps or sample['terminal1']:
                self.endof_episode(sample)
            else:
                obs0 = obs1

            if self.save_step_stats:
                self.step_stats['Training steps'] = self.train_step
                for key in sorted(self.step_stats.keys()):
                    self.logger_step.logkv(key, self.step_stats[key])
                self.logger_step.dumpkvs()

            self.train_step += 1
            self.episode_step += 1
            #print_status("{}/{}".format(self.train_step, self.max_steps))



