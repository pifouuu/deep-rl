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
PARAM = 'memory_SARST_goal_NoGoal_wrapper_NoGoal'
res_files = glob.glob(LOGDIR + PARAM + '/*/' + 'log_step/progress.json')

eval_rewards = [[0]*200 for i in range(len(res_files))]

for j, filename in enumerate(res_files):
    with open(filename, 'r') as json_data:
        lines = json_data.readlines()
        for k, line in enumerate([lines[i] for i in range(999,200000,1000)]) :
            episode_data = json.loads(line)
            if 'Test reward on initial goal' in episode_data:
                eval_rewards[j][k] = episode_data['Test reward on initial goal']


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
ax1.set_xticks(np.arange(0, 200001, 20000))
ax1.set_xlabel("Episode")

ax1.get_yaxis().tick_left()
ax1.tick_params(axis='y', length=0)
ax1.set_ylim(-100, 100)
ax1.set_yticks(np.arange(-100, 100, 20))
ax1.set_ylabel("Reward per episode")

x = range(1000, 200001, 1000)
l = 'solid'
marker = 'o'
label = ''
c = colors[0]
converge = 0
localmin = 0
diverge = 0
for res in eval_rewards:
    final_mean = np.mean(res[140:150])
    final_var = np.var(res[140:150])
    print(final_mean, math.sqrt(final_var))
    if final_mean>90:
        c = colors[1]
        converge+=1
    elif final_mean<10 and final_mean>-10:
        c = colors[2]
        localmin+=1
    elif final_mean<-50:
        c = colors[3]
        diverge+=1
    ax1.plot(x, exp_smooth(res,0.5), linewidth=0.5, color=c, linestyle=l, marker=marker, markersize=0.5, label=label)
legend = ax1.legend(loc=0)
print(converge, localmin, diverge)
plt.show()

