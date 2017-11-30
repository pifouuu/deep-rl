import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import brewer2mpl

def exp_smooth(tab, alpha):
    smooth = [tab[0]]
    for i in range(len(tab)-1):
        smooth.append(alpha*tab[1+i]+(1-alpha)*smooth[i])
    return smooth

LOGDIR = './results/'
PARAMS = ['memory_SARST_goal_NoGoal_wrapper_NoGoal']
param_eval = {}
for param in PARAMS:
    res_files = glob.glob(LOGDIR+param+'/*/'+'log_step/progress.json')
    eval_rewards = [[0] * 1990 for i in range(len(res_files))]
    sum_eval_rewards  = [0]*1990
    for j, filename in enumerate(res_files):
        with open(filename, 'r') as json_data:
            lines = json_data.readlines()
            eval_rewards  = []
            for k, line in enumerate([lines[i] for i in range(1000,200000,100)]) :
                episode_data = json.loads(line)
                if 'Test reward on initial goal' in episode_data:
                    eval_rewards[j][k] = episode_data['Test reward on initial goal']
            # sum_eval_rewards = [x+y for x,y in zip(sum_eval_rewards, eval_rewards)]
    # mean_eval_rewards = [x/len(res_files) for x in sum_eval_rewards]
    # param_eval[param] = mean_eval_rewards

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors
fig = plt.figure()
fig.subplots_adjust(left=0.12, bottom=0.12, right=0.99, top=0.99, wspace=0.1)
ax1 = fig.add_subplot(111)

ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

for spine in ax1.spines.values():
    spine.set_position(('outward', 5))

ax1.tick_params(axis='x', direction='out')
ax1.get_xaxis().tick_bottom()
ax1.set_xlim(0, 200000)
ax1.set_xticks(np.arange(0, 200000, 20000))
ax1.set_xlabel("Episode")

ax1.get_yaxis().tick_left()
ax1.tick_params(axis='y', length=0)
ax1.set_ylim(-10, 100)
ax1.set_yticks(np.arange(-10, 100, 20))
ax1.set_ylabel("Reward per episode")

x = range(1000, 200000, 100)
for res in eval_rewards:
    for key,val in param_eval.items():
        params_names = key.split('_')[0::2]
        params_val = key.split('_')[1::2]
        params_dict = dict(zip(params_names,params_val))
        l = 'solid'
        marker = 'o'
        label = ''
        c = colors[0]
        # if params_dict['goal']=='True':
        #     c = colors[0]
        #     label += 'with goal, '
        # if params_dict['goal']=='False':
        #     c = colors[1]
        #     label += 'without goal, '
        # if params_dict['delta']=='1':
        #     l='dashed'
        #     label += 'clipping 1, '
        # if params_dict['reset']=='False':
        #     marker='*'
        #     label += 'no reset, '

        ax1.plot(x, exp_smooth(val,0.5), linewidth=1, color=c, linestyle=l, marker=marker, markersize=4, label=label)
legend = ax1.legend(loc=0)
plt.show()

# for key,val in param_eval.items():
#     params_names = key.split('_')[0::2]
#     params_val = key.split('_')[1::2]
#     params_dict = dict(zip(params_names,params_val))
#     l = 'solid'
#     marker = 'o'
#     label = ''
#     c = colors[0]
#     # if params_dict['goal']=='True':
#     #     c = colors[0]
#     #     label += 'with goal, '
#     # if params_dict['goal']=='False':
#     #     c = colors[1]
#     #     label += 'without goal, '
#     # if params_dict['delta']=='1':
#     #     l='dashed'
#     #     label += 'clipping 1, '
#     # if params_dict['reset']=='False':
#     #     marker='*'
#     #     label += 'no reset, '
#     x = range(1000,200000,100)
#     ax1.plot(x, exp_smooth(val,0.5), linewidth=1, color=c, linestyle=l, marker=marker, markersize=4, label=label)
# legend = ax1.legend(loc=0)
# plt.show()