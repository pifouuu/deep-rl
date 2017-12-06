import json
import numpy as np
import glob
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import brewer2mpl
import math
import pandas as pd

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

frames = []
keys = []
for num_run, run in enumerate(res_steps):
    df = pd.read_json(run, lines=True)
    df.index = range(200, 200001, 200)
    frames.append(df)
    keys.append(num_run)
expe_res = pd.concat(frames, keys=keys)
expe_res.index.names = ['run', 'step']

means = expe_res.groupby('step').mean()


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