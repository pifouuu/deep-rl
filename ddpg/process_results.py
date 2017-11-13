import json
SUMMARY_DIR = './results/tf_ddpg_tau_0.001_batchsize_64_goal_True_hindsight_False_eval_False_2017_11_13_17_39_32'
filename = SUMMARY_DIR + '/progress.json'

episodes = []
with open(filename, 'r') as json_data:
    lines = json_data.readlines()
    for line in lines:
        episode = json.loads(line)
        episodes.append(episode)
    print(episodes)