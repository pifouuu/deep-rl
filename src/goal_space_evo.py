import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import json_tricks
from keras.models import load_model
import os
from matplotlib.collections import PatchCollection
import numpy as np
Blues = plt.get_cmap('Blues')
import json

log_dir = '../log/cluster0904/ReacherGoal-v0_sarst_final_4_0_10_0.0001_40_0.3_1_1_sparse_prioritized_16_100/20180409223348_116584/'

fig = plt.figure()
ax = plt.axes()
plt.ion()
fig.show()

data = []
with open(log_dir+'log_memory/progress.json','r') as f:
    for line in f:
        data.append(json.loads(line))

ax.set_xlim(left=-0.2, right=0.2)
ax.set_ylim(bottom=-0.2, top=0.2)

# ax.set_xlim(left=-1.2, right=0.6)


param = 'cp'

for i,d in enumerate(data):
    if i % 10 ==0:
        l = d['lines']
        p = d['patches']
        ax.lines.clear()
        ax.collections.clear()
        colors = []
        patch_list = []
        for line_dict in l:
            ax.add_line(lines.Line2D(xdata=line_dict['xdata'],
                                     ydata=line_dict['ydata'],
                                     linewidth=2,
                                     color='blue'))
        for patch_dict in p:
            colors.append(patch_dict['cp'])
            # self.ax.add_patch(patches.Rectangle(xy=patch_dict['angle'],
            #                       width=patch_dict['width'],
            #                       height=patch_dict['height'],
            #                       fill=True,
            #                       facecolor=Blues(color),
            #                       edgecolor=None,
            #                       alpha=0.8))
            patch_list.append(patches.Rectangle(xy=patch_dict['angle'],
                                                width=patch_dict['width'],
                                                height=patch_dict['height'],
                                                fill=True,
                                                edgecolor=None,
                                                alpha=0.8))
        p = PatchCollection(patch_list)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        cb = fig.colorbar(p, ax=ax)
        ax.set_title("Episode {}".format(i))
        plt.pause(0.05)
        cb.remove()
plt.waitforbuttonpress()