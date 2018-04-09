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

log_dir = '../log/local/ReacherGoal-v0_sarst_final_4_0_10_0.0001_5_0.3_1_1_sparse_initial_8_100/20180409122603_956912/log_memory/'

fig = plt.figure()
ax = plt.axes()
plt.ion()
fig.show()

data = []
with open(log_dir+'progress.json') as f:
    for line in f:
        data.append(json.loads(line))

with open(log_dir+'config.txt') as config:
    d = json_tricks.load(config)
    print(d["split_min"])

# ax.set_xlim(left=-0.2, right=0.2)
# ax.set_ylim(bottom=-0.2, top=0.2)

ax.set_xlim(left=-1.2, right=0.6)


param = 'cp'

for i in range(len(video)):
    if i % 1 ==0:
        l = video[i][0]
        p = video[i][1]
        print("iteration ", i)
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