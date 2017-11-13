import json
import numpy as np
SUMMARY_DIR = './results/tf_ddpg_tau_0.001_batchsize_64_goal_True_hindsight_False_eval_False_2017_11_13_17_39_32'
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
        reward.append(episode_data['Reward'])
        qmax_value.append(episode_data['Qmax_value'])
        critic_loss.append(episode_data['Critic_loss'])
        q_mean.append(episode_data['reference_Q_mean'])
        q_std.append(episode_data['reference_Q_std'])
        action_mean.append(episode_data['reference_action_mean'])
        action_std.append(episode_data['reference_action_std'])
        if 'Eval_reward' in episode_data:
            eval_reward.append(episode_data['Eval_reward'])

    print(reward)
    print(eval_reward)