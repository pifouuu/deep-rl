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
PARAMS = ['sarst_final_init_1_inf_linear_False_False',
          'sarst_final_init_1_inf_linear_False_True',
          'sarst_final_init_1_inf_linear_True_False',
          'sarst_final_init_1_inf_linear_True_True',
          'sarst_final_init_1_inf_tanh_False_False',
          'sarst_final_init_1_inf_tanh_False_True',
          'sarst_final_init_1_inf_tanh_True_False',
          'sarst_final_init_1_inf_tanh_True_True',
          'sarst_final_rnd_1_inf_linear_False_False',
          'sarst_final_rnd_1_inf_linear_False_True',
          'sarst_final_rnd_1_inf_linear_True_False',
          'sarst_final_rnd_1_inf_linear_True_True',
          'sarst_final_rnd_1_inf_tanh_False_False',
          'sarst_final_rnd_1_inf_tanh_False_True',
          'sarst_final_rnd_1_inf_tanh_True_False',
          'sarst_final_rnd_1_inf_tanh_True_True']

frames = []
keys = []
for PARAM in PARAMS:
    param_vals = PARAM.split('_')
    param_names = ['memory', 'strategy', 'sampler', 'alpha', 'delta', 'activation', 'invert_grads', 'target_clip']
    res_steps = glob.glob(LOGDIR + PARAM + '/*/' + 'log_steps/progress.json')
    # res_episode = glob.glob(LOGDIR + PARAM + '/*/' + 'log_episodes/progress.json')
    for num_run, run in enumerate(res_steps):
        try:
            df = pd.read_json(run, lines=True)
        except ValueError:
            print("invalid")
        for name, val in zip(param_names, param_vals):
            df[name]=val
        df['num_run']=num_run
        frames.append(df)
expe_res = pd.concat(frames, ignore_index=True)
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