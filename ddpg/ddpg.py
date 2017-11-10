""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp
from baselines import logger
from env_wrapper import GoalContinuousMCWrapper, ContinuousMCWrapper
from memory import Memory, HerMemory
import os
import pickle
import time


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.stats_ops = []
        self.stats_names = []

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

        # Setting up stats
        self.stats_ops += [tf.reduce_mean(self.scaled_out)]
        self.stats_names += ['reference_action_mean']
        self.stats_ops += [reduce_std(self.scaled_out)]
        self.stats_names += ['reference_action_std']

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def get_stats(self, stats_sample):
        actor_values = self.sess.run(self.stats_ops, feed_dict={
            self.inputs: stats_sample['state0'],
        })

        names = self.stats_names[:]
        assert len(names) == len(actor_values)
        stats = dict(zip(names, actor_values))

        return stats


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.stats_ops = []
        self.stats_names = []

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

        # Setting up stats
        self.stats_ops += [tf.reduce_mean(self.out)]
        self.stats_names += ['reference_Q_mean']
        self.stats_ops += [reduce_std(self.out)]
        self.stats_names += ['reference_Q_std']


    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        #net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_stats(self, stats_sample):
        critic_values = self.sess.run(self.stats_ops, feed_dict={
            self.inputs: stats_sample['state0'],
            self.action: stats_sample['action'],
        })

        names = self.stats_names[:]
        assert len(names) == len(critic_values)
        stats = dict(zip(names, critic_values))

        # critic_with_actor_values = self.sess.run(self.stats_ops, feed_dict={
        #     self.inputs: stats_sample[0],
        #     self.action: stats_sample['action'],
        # })
        #
        # for name, val in zip(names, critic_with_actor_values):
        #     stats[name+'_actor'] = val

        return stats

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

# def build_summaries():
#     episode_reward = tf.Variable(0.)
#     tf.summary.scalar("Reward", episode_reward)
#     episode_ave_max_q = tf.Variable(0.)
#     tf.summary.scalar("Qmax Value", episode_ave_max_q)
#
#     summary_vars = [episode_reward, episode_ave_max_q]
#     summary_ops = tf.summary.merge_all()
#
#     return summary_ops, summary_vars

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))




# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise, memory, env_wrapper):

    # Set up summary Ops
    # summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Init stats sample
    stats_sample = None

    epoch_start_time = time.time()
    for i in range(int(args['max_episodes'])):

        obs = env.reset()

        # Selects a goal for the current episode
        goal_episode = env_wrapper.sample_goal(obs)

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            state0 = env_wrapper.process_observation(obs, goal_episode)

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            action = actor.predict(np.reshape(state0, (1, actor.s_dim))) + actor_noise()

            new_obs, r_env, done_env, info = env.step(action[0])
            buffer_item = env_wrapper.process_step(state0, goal_episode, action, new_obs, r_env, done_env, info)
            memory.append(buffer_item)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if memory.nb_entries > int(args['minibatch_size']):
                sample = memory.sample(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    sample['state1'], actor.predict_target(sample['state1']))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if sample['terminal1'][k]:
                        y_i.append(sample['reward'][k])
                    else:
                        y_i.append(sample['reward'][k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    sample['state0'], sample['action'], np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(sample['state0'])
                grads = critic.action_gradients(sample['state0'], a_outs)
                actor.train(sample['state0'], grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            obs = new_obs
            ep_reward += buffer_item['reward']

            if done_env or buffer_item['terminal1']:
                if stats_sample is None:
                    # Get a sample and keep that fixed for all further computations.
                    # This allows us to estimate the change in value for the same set of inputs.
                    stats_sample = memory.sample(batch_size=int(args['minibatch_size']))

                combined_stats = {}
                actor_stats = actor.get_stats(stats_sample)
                for key in sorted(actor_stats.keys()):
                    combined_stats[key] = (actor_stats[key])
                critic_stats = critic.get_stats(stats_sample)
                for key in sorted(critic_stats.keys()):
                    combined_stats[key] = (critic_stats[key])
                combined_stats['Reward'] = ep_reward
                combined_stats['Qmax value'] = ep_ave_max_q / float(j)

                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')
                logdir = logger.get_dir()
                if logdir:
                    if hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                            pickle.dump(env.get_state(), f)

                # summary_str = sess.run(summary_ops, feed_dict={
                #     summary_vars[0]: ep_reward,
                #     summary_vars[1]: ep_ave_max_q / float(j)
                # })
                #
                # writer.add_summary(summary_str, i)
                # writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Duration: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j)), time.time() - epoch_start_time))
                break

def main(args):
    dirname = '_tau_'+str(args['tau'])+'batchsize_'+str(args['minibatch_size'])+'goal_'+str(args['with_goal'])+'hindsight_'+str(args['with_hindsight'])
    dir = args['summary_dir']+dirname
    logger.configure(dir=dir,format_strs=['stdout', 'json', 'tensorboard'])
    #logger.configure(dir=args['summary_dir'],format_strs=['stdout'])


    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        if args['with_goal']:
            env_wrapper = GoalContinuousMCWrapper()
        else:
            env_wrapper = ContinuousMCWrapper()

        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.shape[0]
        state_dim = env_wrapper.state_shape[0]
        action_dim = env_wrapper.action_shape[0]

        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # Initialize replay memory
        if args['with_hindsight']:
            memory = HerMemory(env_wrapper, with_reward=True, limit=int(1e6), strategy='last')
        else:
            memory = Memory(env_wrapper, with_reward=True, limit=int(1e6))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise, memory, env_wrapper)

        if args['use_gym_monitor']:
            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)
    parser.add_argument('--with-goal', help='concatenate goal and observation in states', action='store_true')
    parser.add_argument('--with-hindsight', help='use hindsight experience replay', action='store_true')

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=0)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=500)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    parser.set_defaults(with_goal=True)
    parser.set_defaults(with_hindsight=False)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
