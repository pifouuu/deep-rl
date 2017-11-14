import json
import numpy as np
SUMMARY_DIR = './results/tf_ddpg_tau_0.001_batchsize_64_goal_True_hindsight_True_eval_True_2017_11_14_14_07_37'
filename = SUMMARY_DIR + '/progress.json'

reward = []
qmax_value = []
critic_loss = []
q_mean = []
q_std = []
action_mean = []
action_std = []
eval_reward = []

with open(filename, 'r') as json_data:
    lines = json_data.readlines()
    episode = 0
    for line in lines:

        episode_data = json.loads(line)

        episode = episode_data['episode']
        env_step = episode_data['Env steps']
        train_step = episode_data['Train steps']

        if 'Eval_reward' in episode_data:
            eval_reward.append((episode, env_step, train_step, episode_data['Eval_reward']))
        if 'Reward' in episode_data:
            reward.append((episode, env_step, train_step, episode_data['Reward']))
        if 'Qmax_value' in episode_data:
            qmax_value.append((episode, env_step, train_step, episode_data['Qmax_value']))
        if 'reference_Q_mean' in episode_data:
            q_mean.append((episode, env_step, train_step, episode_data['reference_Q_mean']))
        if 'reference_Q_std' in episode_data:
            q_std.append((episode, env_step, train_step, episode_data['reference_Q_std']))
        if 'Critic_loss' in episode_data:
            critic_loss.append((episode, env_step, train_step, episode_data['Critic_loss']))
        if 'reference_action_mean' in episode_data:
            action_mean.append((episode, env_step, train_step, episode_data['reference_action_mean']))
        if 'reference_action_std' in episode_data:
            action_std.append((episode, env_step, train_step, episode_data['reference_action_std']))



    print(reward)
    print(eval_reward)