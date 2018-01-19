import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import brewer2mpl
import math

def exp_smooth(tab, alpha):
    smooth = [tab[0]]
    for i in range(len(tab)-1):
        smooth.append(alpha*tab[1+i]+(1-alpha)*smooth[i])
    return smooth

LOGDIR = './results/'
PARAM = 'm_sarst_g_goalC_alpha_0.9_w_goalC_tclip'
res_step = glob.glob(LOGDIR + PARAM + '/*/' + 'log_step/progress.json')
res_steps = glob.glob(LOGDIR + PARAM + '/*/' + 'log_steps/progress.json')

res_episode = glob.glob(LOGDIR + PARAM + '/*/' + 'log_episodes/progress.json')

results_steps = {}
results_episodes = {}

for j, filename in enumerate(res_steps):
    with open(filename, 'r') as json_data:
        lines = json_data.readlines()
        # read_freq = 1000
        for k, line in enumerate(lines):
        # for k, line in enumerate([lines[i] for i in range(read_freq-1,200000,read_freq)]) :
            episode_data = json.loads(line)
            # N = int(200000/read_freq)
            N=1000
            for key, val in episode_data.items():
                if key not in results_steps:
                    results_steps[key] = [[0]*N for _ in range(len(res_steps))]
                results_steps[key][j][k] = val

for j, filename in enumerate(res_episode):
    with open(filename, 'r') as json_data:
        lines = json_data.readlines()
        for k, line in enumerate(lines):
            episode_data = json.loads(line)
            for key, val in episode_data.items():
                if key not in results_episodes:
                    results_episodes[key] = [[] for _ in range(len(res_episode))]
                results_episodes[key][j].append(val)

bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors

for key, val in results_steps.items():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Step")
    ax1.set_ylabel(key)
    x = range(200, 200001, 200)
    for res in val:
        ax1.plot(x, exp_smooth(res,0.5))
        ax1.set_title(PARAM)
    ax1.legend()
    plt.savefig(LOGDIR+PARAM+'/'+key.replace(' ', '_')+'.png')

for key, val in results_episodes.items():
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(key)
    for num_run, res in enumerate(val):
        x = results_episodes['Episode'][num_run]
        ax1.plot(x, exp_smooth(res,0.5))
        ax1.set_title(PARAM)
    ax1.legend()
    plt.savefig(LOGDIR+PARAM+'/'+key.replace(' ', '_')+'.png')