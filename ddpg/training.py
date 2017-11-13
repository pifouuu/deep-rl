import tensorflow as tf
import numpy as np

import logger
import os
import pickle
import time

def log(stats):
    for key in sorted(stats.keys()):
        logger.record_tabular(key, stats[key])
    logger.dump_tabular()


def train(sess, env, eval_env, args, actor, critic, actor_noise, memory, env_wrapper):

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
    obs = env.reset()

    for i in range(int(args['max_episodes'])):

        # Selects a goal for the current episode
        goal_episode = env_wrapper.sample_goal(obs)
        init_state = obs

        ep_reward = 0
        ep_ave_max_q = 0
        critic_losses = []

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
                predicted_q_value, critic_loss, _ = critic.train(
                    sample['state0'], sample['action'], np.reshape(y_i, (int(args['minibatch_size']), 1)))
                critic_losses.append(critic_loss)

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

            if buffer_item['terminal1']:
                obs = env.reset()
                break

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
        combined_stats['Qmax_value'] = ep_ave_max_q / float(j)
        combined_stats['Critic_loss'] = np.mean(critic_losses)

        print('| Reward: {:d} | Episode: {:d} | Init_state: {:.2f} | Goal: {:.2f} | Qmax: {:.4f} | Duration: {:.4f}'.format(int(ep_reward), \
                                                                                        i, init_state[0], goal_episode[0],
                                                                                        (ep_ave_max_q / float(j)),
                                                                                        time.time() - epoch_start_time))

        if args['eval'] and i % int(args['eval_freq']) == 0 and i>0:
            ep_eval_reward = 0
            eval_obs = eval_env.reset()
            eval_goal = env_wrapper.sample_eval_goal()

            for k in range(int(args['eval_steps'])):

                eval_state0 = env_wrapper.process_observation(eval_obs, eval_goal)

                eval_action = actor.predict(np.reshape(eval_state0, (1, actor.s_dim)))

                new_eval_obs, r_eval_env, done_eval_env, info_eval = eval_env.step(eval_action[0])
                eval_buffer_item = env_wrapper.process_step(eval_state0, eval_goal, eval_action, new_eval_obs,
                                                       r_eval_env, done_eval_env, info_eval)

                ep_eval_reward += eval_buffer_item['reward']

                if eval_buffer_item['terminal1']:
                    break
                else:
                    eval_obs = new_eval_obs
            combined_stats['Eval_reward'] = ep_eval_reward

        log(combined_stats)