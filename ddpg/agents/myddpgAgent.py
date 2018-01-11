
import numpy as np

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
        self.best_score = 8600


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

    def train_critic_no_averaging(self, samples):

        # Calculate targets
        target_q = self.critic.predict_target(samples['state1'], self.actor.predict_target(samples['state1']))
        critic_loss = 0
        for k in range(self.batch_size):
            y_i = []
            if samples['terminal1'][k]:
                y_i.append(samples['reward'][k])
            else:
                y_i.append(samples['reward'][k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            s0 = np.array([samples['state0'][k]])
            a0 = np.array([samples['action'][k]])
            critic_loss += self.critic.train(s0, a0, y_i)

        critic_loss = critic_loss/self.batch_size
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
        sample['reward'] = reward_env #patch OSD to deal with mujoco simulations
        sample['terminal1'] = done_env #patch OSD to deal with mujoco simulations
        return obs1, sample

    def train(self):
        samples_train = self.memory.sample(self.batch_size)
        if (self.averaging):
            self.train_critic(samples_train)
        else:
            self.train_critic_no_averaging(samples_train)
        self.train_actor(samples_train)
        self.update_targets()


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
                #print_status("{}/{}".format(k, self.max_episode_steps))
            test_rewards.append(ep_test_reward)

        mean_reward = np.mean(test_rewards)
        self.episode_stats['New Training steps'] = self.train_step
        self.episode_stats['New Test reward'] = mean_reward
        self.step_stats['Test reward'] = mean_reward
        print ('Training steps', self.train_step, ': reward', mean_reward)
        if mean_reward> self.best_score:
            self.actor.save_target_weights("actors/good_actor_{}.save".format(mean_reward),overwrite=True)
            self.best_score = mean_reward
            #portrait_actor(self.actor.target_model,self.test_env,save_figure=True)
            #self.actor.print_target_weights()
            #sys.exit()

        for key in sorted(self.episode_stats.keys()):
            self.logger_episode.logkv(key, self.episode_stats[key])
        self.logger_episode.dumpkvs()
        return mean_reward

    def endof_episode(self, sample):
        '''
        self.episode_stats['Episode'] = self.episode
        self.episode_stats['Start'] = self.episode_init[0]
        self.episode_stats['Goal'] = self.train_goal[0]
        self.episode_stats['Train reward'] = self.episode_reward
        self.episode_stats['Episode steps'] = self.episode_step
        self.episode_stats['Training steps'] = self.train_step
        #self.episode_stats['Episode difficulty'] = difficulty
        self.episode_stats['Goal reached'] = self.nb_goals_reached

 
        for key in sorted(self.episode_stats.keys()):
            self.logger_episode.logkv(key, self.episode_stats[key])
        self.logger_episode.dumpkvs()
        '''

        self.train_env.reset()
        self.episode_step = 0
        self.episode_reward = 0
        self.episode += 1
        if sample['terminal1']: self.nb_goals_reached += 1

    def run(self):

        # Initialize target network weights
        self.update_targets()

        #print("r6:", np.random.random())

        obs0 = self.train_env.reset()
        self.episode_init = obs0

        #TODO : pass on to a sample goal function in the agent, not in the wrapper
        self.train_goal = self.env_wrapper.sample_goal()

        #print("r7:", np.random.random())

        while self.train_step < self.max_steps:

            obs1, sample = self.step(obs0, self.train_goal, test=False)

            self.memory.append(sample)

            reward = sample['reward']
            self.episode_reward += reward
            self.total_reward += reward

            if self.memory.nb_entries > 3*self.batch_size:
                self.train()

            #print("r8:", np.random.random())

            if self.train_step % self.eval_freq == 0:
                self.test()

            #print("r9:", np.random.random())

            if self.episode_step >= self.max_episode_steps or sample['terminal1']:
                self.endof_episode(sample)
            else:
                obs0 = obs1

            if self.save_step_stats:
                self.step_stats['Training steps'] = self.train_step
                for key in sorted(self.step_stats.keys()):
                    self.logger_step.logkv(key, self.step_stats[key])
                self.logger_step.dumpkvs()

            #print("r10:", np.random.random())

            self.train_step += 1
            self.episode_step += 1
            #print_status("{}/{}".format(self.train_step, self.max_steps))



