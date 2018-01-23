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
PARAMS = ['memory_SARST_goal_IntervalCurri_wrapper_IntervalCurri',
          # 'memory_SARST_goal_NoGoal_wrapper_NoGoal',
          # 'memory_SARST_goal_Random_wrapper_WithGoal',
          # 'memory_SARST_goal_GoalCurri_wrapper_GoalCurri',
          # 'memory_hindsight_SARST_goal_GoalCurri_wrapper_GoalCurri_strategy_future'
          ]

# brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
colors = bmap.mpl_colors
fig = plt.figure()
fig.subplots_adjust(left=0.12, bottom=0.12, right=0.99, top=0.99, wspace=0.1)

param_eval = {}
for i, param in enumerate(PARAMS):
    print(param)

    ax1 = fig.add_subplot(32*10+i+1)

    ax1.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    for spine in ax1.spines.values():
        spine.set_position(('outward', 5))

    res_files = glob.glob(LOGDIR+param+'/*/'+'log_step/progress.json')
    if len(res_files) != 0:
        eval_rewards = [[0] * 200 for i in range(len(res_files))]
        x = range(1000, 200001, 1000)
        for j, filename in enumerate(res_files):
            with open(filename, 'r') as json_data:
                lines = json_data.readlines()
                for k, line in enumerate([lines[i] for i in range(999, 200000, 1000)]):
                    episode_data = json.loads(line)
                    if 'Test reward on initial goal_wrappers' in episode_data:
                        eval_rewards[j][k] = episode_data['Test reward on initial goal_wrappers']

    if len(res_files) == 0:
        res_files = glob.glob(LOGDIR+param+'/*/'+'log_steps/progress.json')
        eval_rewards = [[0] * 1000 for _ in range(len(res_files))]
        x = range(200, 200001, 200)

        for j, filename in enumerate(res_files):
            with open(filename, 'r') as json_data:
                lines = json_data.readlines()
                print(len(lines))
                for k, line in enumerate(lines) :
                    episode_data = json.loads(line)
                    if 'Test reward on initial goal_wrappers' in episode_data:
                        eval_rewards[j][k] = episode_data['Test reward on initial goal_wrappers']

    #         sum_eval_rewards = [x+y for x,y in zip(sum_eval_rewards, eval_rewards)]
    # mean_eval_rewards = [x/len(res_files) for x in sum_eval_rewards]
    # param_eval[param] = mean_eval_rewards


    ax1.tick_params(axis='x', direction='out')
    ax1.get_xaxis().tick_bottom()
    ax1.set_xlim(0, 200000)
    ax1.set_xticks(np.arange(0, 200000, 20000))
    ax1.set_xlabel("Episode")

    ax1.get_yaxis().tick_left()
    ax1.tick_params(axis='y', length=0)
    ax1.set_ylim(-100, 100)
    ax1.set_yticks(np.arange(-100, 100, 20))
    ax1.set_ylabel("Reward per episode")

    converge = 0
    localmin = 0
    diverge = 0
    c = colors[0]

    for res in eval_rewards:
        final_mean = np.mean(res[-20:])
        final_var = np.var(res[-20:])
        l = 'solid'
        marker = 'o'
        label = ''
        if final_mean > 90:
            c = colors[1]
            converge += 1
        elif final_mean < 5 and final_mean > -5:
            c = colors[2]
            localmin += 1
        elif final_mean < -15:
            c = colors[3]
            diverge += 1
        ax1.plot(x, exp_smooth(res,0.5), linewidth=0.5, color=c, linestyle=l, marker=marker, markersize=0.5, label=label)
        ax1.set_title(param)
    legend = ax1.legend(loc=0)

    print(converge, localmin, diverge)

plt.show()

# for key,val in param_eval.items():
#     params_names = key.split('_')[0::2]
#     params_val = key.split('_')[1::2]
#     params_dict = dict(zip(params_names,params_val))
#     l = 'solid'
#     marker = 'o'
#     label = ''
#     c = colors[0]
#     # if params_dict['goal_wrappers']=='True':
#     #     c = colors[0]
#     #     label += 'with goal_wrappers, '
#     # if params_dict['goal_wrappers']=='False':
#     #     c = colors[1]
#     #     label += 'without goal_wrappers, '
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