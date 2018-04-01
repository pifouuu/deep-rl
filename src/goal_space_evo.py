import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import json_tricks
from keras.models import load_model
import os

Blues = plt.get_cmap('Blues')

log_dir = '../log/cluster0104/CMCPos-v0_sarst_final_0_10_0_50_0.3_1_1_sparse_random_16_1000/20180329201645_539505/'

fig = plt.figure()
ax = plt.axes()
plt.ion()
fig.show()

with open(log_dir+'pickle/comp_progress.pkl','rb') as handle:
    video = pickle.load(handle)

with open(log_dir+'config.txt') as config:
    d = json_tricks.load(config)
    print(d["split_min"])
print(len(video))

# ax.set_xlim(left=-0.2, right=0.2)
# ax.set_ylim(bottom=-0.2, top=0.2)

ax.set_xlim(left=-1.2, right=0.6)


param = 'cp'

for i in range(len(video)):
    if i % 20 ==0:
        l = video[i][0]
        p = video[i][1]
        print("iteration ", i)
        ax.lines.clear()
        ax.patches.clear()
        for line_dict in l:
            ax.add_line(lines.Line2D(xdata=line_dict['xdata'],
                                     ydata=line_dict['ydata'],
                                     linewidth=2,
                                     color='blue'))
        for patch_dict in p:
            if patch_dict['max_'+param] - patch_dict['min_'+param] == 0:
                color = 0
            else:
                color = (patch_dict[param] - patch_dict['min_'+param]) / (patch_dict['max_'+param] - patch_dict['min_'+param])
            # color = patch_dict[param]/ patch_dict['max_'+param]
            # print("color ", color)
            ax.add_patch(patches.Rectangle(xy=patch_dict['angle'],
                                      width=patch_dict['width'],
                                      height=patch_dict['height'],
                                      fill=True,
                                      facecolor=Blues(color),
                                      edgecolor=None,
                                      alpha=0.8))
        ax.set_title("Episode {}".format(i))
        plt.pause(0.05)
plt.waitforbuttonpress()