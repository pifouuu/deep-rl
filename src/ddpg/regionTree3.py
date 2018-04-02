from collections import deque
from gym.spaces import Box
import numpy as np
import itertools
from ddpg.memory import SARSTMemory, EpisodicHerSARSTMemory
import os


import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib import animation
Blues = plt.get_cmap('Blues')

class Region(Box):

    def __init__(self, low = np.array([-np.inf]), high=np.array([np.inf]), maxlen=0, n_window=0, dims=None):
        super(Region, self).__init__(low, high)
        self.maxlen = maxlen
        self.points = deque(maxlen=self.maxlen)
        self.n_window = n_window
        self.CP = 0
        self.competence = 0
        self.max_competence = 0
        self.min_competence = 0
        self.sum_competence = 0
        self.max_CP = 0
        self.min_CP = 0
        self.sum_CP = 0
        self.dim_split = None
        self.val_split = None
        self.dims = dims
        self.freq = 0

    def sample(self):
        return np.random.uniform(low=self.low[self.dims], high=self.high[self.dims])

    def contains(self, x):
        return x.shape == self.low[self.dims].shape and (x >= self.low[self.dims]).all() and (x <= self.high[self.dims]).all()

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left, maxlen=self.maxlen, n_window = self.n_window, dims=self.dims)
        right = Region(low_right, self.high, maxlen=self.maxlen, n_window = self.n_window, dims=self.dims)
        left.CP = self.CP
        right.CP = self.CP
        left.competence = self.competence
        right.competence = self.competence
        for point in self.points:
            if left.contains(point[0]):
                left.points.append(point)
            else:
                right.points.append(point)
        left.update_CP()
        right.update_CP()
        return left, right

    def update_CP(self):
        if self.size > 2*self.n_window:
            len = self.size
            q1 = [pt[1] for pt in list(itertools.islice(self.points, len-self.n_window, len))]
            q2 = [pt[1] for pt in list(itertools.islice(self.points, len-2*self.n_window, len-self.n_window))]
            self.CP = 1/2 + (np.sum(q1)-np.sum(q2))/(2*self.n_window)
            self.competence = (np.sum(q1)+np.sum(q2))/(2*self.n_window)
        self.max_CP = self.CP
        self.max_competence = self.competence
        self.min_competence = self.competence
        self.sum_CP = self.CP
        self.min_CP = self.CP
        assert self.CP >= 0

    @property
    def size(self):
        return len(self.points)

    @property
    def is_leaf(self):
        return (self.dim_split is None)

    @property
    def full(self):
        return self.size == self.maxlen

class TreeMemory():
    def __init__(self, space, dims, buffer, actor, critic, max_regions, n_split, split_min, alpha, maxlen, n_window, render, sampler):
        self.n_split = n_split
        self.split_min = split_min
        self.maxlen = maxlen
        self.n_window = n_window
        self.max_regions = max_regions
        self.alpha = alpha
        self.dims = dims
        self.figure_dims = dims

        self.buffer = buffer
        self.sampler = sampler

        self.ax = None
        self.figure = None
        self.lines = []
        self.patches = []
        self.points = []
        self.history = []

        capacity = 1
        while capacity < max_regions:
            capacity *= 2
        self.capacity = capacity
        self.region_array = [Region() for _ in range(2 * self.capacity)]
        self.region_array[1] = Region(space.low, space.high, maxlen=self.maxlen, n_window=self.n_window, dims=dims)
        self.update_CP_tree()
        self.n_leaves = 1
        self.render = render
        self.actor = actor
        self.critic = critic

        self.minval = self.buffer.env.reward_range[0] / (1 - 0.99)
        self.maxval = self.buffer.env.reward_range[1]

    def end_episode(self, goal_reached):
        self.buffer.end_episode(goal_reached)
        if self.buffer.env.goal_parameterized:
            self.sample_competence()
            # regions = self.find_regions(self.buffer.env.goal)
            # region_idx = np.random.choice(regions)
            # starts = [self.buffer.env.get_start() for _ in range(1)]
            self.update_tree()
            self.update_CP_tree()
            self.update_display()

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def append(self, buffer_item):
        self.buffer.append(buffer_item)

    def build_exp(self, state, action, next_state, reward, terminal):
        return self.buffer.build_exp(state, action, next_state, reward, terminal)

    def sample_competence(self):
        self._sample_competence(1)

    def _sample_competence(self, idx):
        region = self.region_array[idx]
        if region.is_leaf:
            region_goals = [region.sample() for _ in range(1)]
            starts = [self.buffer.env.get_start() for _ in range(1)]
            states = np.array([np.hstack([start,goal]) for start,goal in zip(starts, region_goals)])
            a_outs = self.actor.predict(states)
            q_outs = list(self.critic.predict(states, a_outs))
            corr_vals = [(val - self.minval) / (self.maxval - self.minval) for val in q_outs]
            for goal, val in zip(region_goals, corr_vals):
                region.points.append((goal,val))
        else:
            self._sample_competence(2 * idx)
            self._sample_competence(2 * idx + 1)

    def update_display(self):
        self.compute_image()
        if self.render:
            if self.figure is None:
                self.init_display()
            self.plot_image()

        self.history.append([self.lines.copy(), self.patches.copy()])

    def divide(self, n):
        assert n & (n-1) == 0 #n must be a power of 2
        self._divide(1, n, 0)

    def _divide(self, idx , n, dim_idx):
        if n > 1:
            dim = self.dims[dim_idx]
            region = self.region_array[idx]
            low = region.low[dim]
            high = region.high[dim]
            val_split = (high+low)/2
            self.region_array[2 * idx], self.region_array[2 * idx + 1] = region.split(dim, val_split)
            region.dim_split = dim
            region.val_split = val_split
            self.n_leaves += 1
            next_dim_idx = (dim_idx+1)%(len(self.dims))
            self._divide(2 * idx, n/2, next_dim_idx)
            self._divide(2 * idx + 1, n/2, next_dim_idx)

    def find_regions(self, sample):
        regions = self._find_regions(sample, 1)
        return regions

    def _find_regions(self, sample, idx):
        regions = [idx]
        region = self.region_array[idx]
        if not region.is_leaf:
            left = self.region_array[2 * idx]
            if left.contains(sample):
                regions_left = self._find_regions(sample, 2 * idx)
                regions += regions_left
            else:
                regions_right = self._find_regions(sample, 2 * idx + 1)
                regions += regions_right
        return regions

    def update_tree(self):
        self._update_tree(1)

    def _update_tree(self, idx):
        region = self.region_array[idx]
        region.update_CP()
        if not region.is_leaf:
            self._update_tree(2 * idx)
            self._update_tree(2 * idx + 1)
        else:
            if region.full and idx < self.capacity:
                self.split(idx)

    def update_CP_tree(self):
        self._update_CP_tree(1)

    def _update_CP_tree(self, idx):
        region = self.region_array[idx]
        if region.is_leaf:
            region.max_CP = region.CP
            region.min_CP = region.CP
            region.sum_CP = region.CP
            region.max_competence = region.competence
            region.min_competence = region.competence
            region.sum_competence = region.competence

        else:
            left = self.region_array[2 * idx]
            right = self.region_array[2 * idx + 1]
            split_eval = self.split_eval(left, right)
            to_merge = left.is_leaf and right.is_leaf and split_eval < self.split_min
            if to_merge:
                region.dim_split = None
                region.val_split = None
                self.region_array[2 * idx] = None
                self.region_array[2 * idx + 1] = None
                self.n_leaves -= 1
                print('merge')
                self._update_CP_tree(idx)
            else:
                self._update_CP_tree(2 * idx)
                self._update_CP_tree(2 * idx + 1)
                region.max_CP = np.max([left.max_CP, right.max_CP])
                region.min_CP = np.min([left.min_CP, right.min_CP])
                region.sum_CP = np.sum([left.sum_CP, right.sum_CP])
                region.max_competence = np.max([left.max_competence, right.max_competence])
                region.min_competence = np.min([left.min_competence, right.min_competence])
                region.sum_competence = np.sum([left.sum_competence, right.sum_competence])

    def split_eval(self, left, right):
        return left.size * right.size * np.sqrt((right.CP-left.CP)**2)

    def split(self, idx):
        eval_splits = []
        if self.n_leaves < self.max_regions:
            region = self.region_array[idx]
            for dim in self.dims:
                sub_regions = np.linspace(region.low[dim], region.high[dim], self.n_split+2)
                for num_split, split_val in enumerate(sub_regions[1:-1]):
                    temp_left, temp_right = region.split(dim, split_val)
                    eval_splits.append(self.split_eval(temp_left, temp_right))
            width = np.max(eval_splits)-np.min(eval_splits)
            if width != 0:
                split_idx = np.argmax(eval_splits)
                if eval_splits[split_idx] > self.split_min:
                    region.dim_split = self.dims[split_idx // self.n_split]
                    region.val_split = np.linspace(region.low[region.dim_split], region.high[region.dim_split], self.n_split+2)[split_idx % self.n_split+1]
                    self.region_array[2 * idx], self.region_array[2 * idx + 1] = region.split(region.dim_split, region.val_split)
                    print('splint succeeded: dim=', region.dim_split,
                          ' val=', region.val_split,
                          ' diff=', eval_splits[split_idx])
                    self.n_leaves += 1

    def compute_image(self, with_points=False):
        self.lines.clear()
        self.patches.clear()
        self.points.clear()
        self._compute_image(1, with_points)
        print("max_cp : ", self.max_CP)

    def _compute_image(self, idx, with_points=False):
        region = self.region_array[idx]
        if len(self.figure_dims) > 1:
            low1 = region.low[self.figure_dims[1]]
            high1 = region.high[self.figure_dims[1]]
        else:
            low1 = 0
            high1 = 1

        if region.is_leaf:
            angle = (region.low[self.figure_dims[0]], low1)
            width = region.high[self.figure_dims[0]] - region.low[self.figure_dims[0]]
            height = high1 - low1
            # print("region ", idx, " cp : ", region.CP)
            self.patches.append({'angle': angle,
                                 'width': width,
                                 'height': height,
                                 'max_cp': self.max_CP,
                                 'min_cp': self.min_CP,
                                 'cp': region.CP,
                                 'max_competence': self.max_competence,
                                 'min_competence': self.min_competence,
                                 'competence': region.competence,
                                 'freq': region.freq
                                 })
            if with_points:
                for point in region.points:
                    self.points.append(point)
        else:
            if region.dim_split == self.figure_dims[0]:
                line1_xs = 2 * [region.val_split]
                line1_ys = [low1, high1]
                self.lines.append({'xdata': line1_xs,
                                   'ydata': line1_ys})
            elif len(self.figure_dims)>1 and region.dim_split == self.figure_dims[1]:
                line1_ys = 2 * [region.val_split]
                line1_xs = [region.low[self.figure_dims[0]], region.high[self.figure_dims[0]]]
                self.lines.append({'xdata': line1_xs,
                               'ydata': line1_ys})

            self._compute_image(2 * idx, self.figure_dims)
            self._compute_image(2 * idx + 1, self.figure_dims)

    def init_display(self):
        self.figure = plt.figure()
        self.ax = plt.axes()
        self.ax.set_xlim(self.root.low[self.figure_dims[0]], self.root.high[self.figure_dims[0]])
        if len(self.figure_dims)>1:
            self.ax.set_ylim(self.root.low[self.figure_dims[1]], self.root.high[self.figure_dims[1]])
        else:
            self.ax.set_ylim(0, 1)
        plt.ion()
        plt.show()

    def plot_image(self, with_points=False):
        self.ax.lines.clear()
        self.ax.patches.clear()
        for line_dict in self.lines:
            self.ax.add_line(lines.Line2D(xdata=line_dict['xdata'],
                                          ydata=line_dict['ydata'],
                                          linewidth=2,
                                          color='blue'))
        for patch_dict in self.patches:
            if patch_dict['max_cp'] - patch_dict['min_cp'] == 0:
                color = 0
            else:
                color = (patch_dict['cp']-patch_dict['min_cp'])/(patch_dict['max_cp']-patch_dict['min_cp'])
                # color = region.competence/self.max_competence
            self.ax.add_patch(patches.Rectangle(xy=patch_dict['angle'],
                                  width=patch_dict['width'],
                                  height=patch_dict['height'],
                                  fill=True,
                                  facecolor=Blues(color),
                                  edgecolor=None,
                                  alpha=0.8))
        # if with_points:
        #     x, y, z = zip(*[(point.pos[0], point.pos[1], point.val) for point in self.points])
        #     sizes = [0.01 + ze for ze in z]
        #     self.ax.scatter(x, y, s=sizes, c='red')
        plt.draw()
        plt.pause(0.001)


    def find_prop_region(self, sum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= sum
        """
        assert 0 <= sum <= self.sum_CP + 1e-5
        idx = 1
        while not self.region_array[idx].is_leaf:
            s = self.region_array[2 * idx].sum_CP
            if s > sum:
                idx = 2 * idx
            else:
                sum -= s
                idx = 2 * idx + 1
        return self.region_array[idx]

    def sample_prop(self):
        sum = self.sum_CP
        mass = np.random.random() * sum
        region = self.find_prop_region(mass)
        region.freq += 1
        sample = region.sample()
        return sample

    def sample_random(self):
        sample = self.root.sample()
        regions = self.find_regions(sample)
        self.region_array[regions[-1]].freq += 1
        return sample

    def sample_goal(self):
        if self.sampler=='prioritized':
            p = np.random.random()
            if p < self.alpha:
                return self.sample_random()
            else:
                return self.sample_prop()
        elif self.sampler=='random':
            return self.sample_random()
        elif self.sampler=='initial':
            return self.buffer.env.initial_goal
        else:
            raise RuntimeError

    @property
    def root(self):
        return self.region_array[1]

    @property
    def max_CP(self):
        return self.root.max_CP

    @property
    def max_competence(self):
        return self.root.max_competence

    @property
    def min_competence(self):
        return self.root.min_competence

    @property
    def sum_competence(self):
        return self.root.sum_competence

    @property
    def min_CP(self):
        return self.root.min_CP

    @property
    def sum_CP(self):
        return self.root.sum_CP

# class zones():
#     def __init__(self):
#
#         self.zones = []
#         self.samples_per_zone = []
#         self.comp_per_zone = []
#         self.zone_difficulties = []
#
#     def compute_comp(self, goal):
#         goal_zone = None
#         i = 0
#         while goal_zone is None:
#             if self.zones[i].contains(goal):
#                 goal_zone = self.zones[i]
#             i += 1
#         n_samples = self.samples_per_zone[i - 1]
#
#         if n_samples < self.zone_difficulties[i - 1]:
#             comp = 0
#         else:
#             comp = np.min([1, (n_samples - self.zone_difficulties[i - 1]) / 1000])
#
#         self.samples_per_zone[i - 1] += 1
#         return comp
#
# class zones1(zones):
#     def __init__(self):
#         super(zones1, self).__init__()
#         zone1 = Region(np.array([-1.2, -0.07]), np.array([-0.6, 0]))
#         zone2 = Region(np.array([-0.6, -0.07]), np.array([0, 0]))
#         zone3 = Region(np.array([0, -0.07]), np.array([0.6, 0]))
#         zone4 = Region(np.array([-1.2, 0]), np.array([-0.6, 0.07]))
#         zone5 = Region(np.array([-0.6, 0]), np.array([0, 0.07]))
#         zone6 = Region(np.array([0, 0]), np.array([0.6, 0.07]))
#         self.zones = [zone1, zone2, zone3, zone4, zone5, zone6]
#         self.samples_per_zone = 6 * [0]
#         self.comp_per_zone = 6 * [0]
#         self.zone_difficulties = [0, 200, 400, 800, 1000, 1200]
#
# class zones2(zones):
#     def __init__(self):
#         super(zones2, self).__init__()
#         zone1 = Region(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
#         self.zones = [zone1]
#         self.samples_per_zone = [0]
#         self.comp_per_zone = [0]
#         self.zone_difficulties = [200]
#
# class zones3(zones):
#     def __init__(self):
#         super(zones3, self).__init__()
#         zone1 = Region(np.array([-1.2]), np.array([0]))
#         zone2 = Region(np.array([0]), np.array([0.6]))
#         self.zones = [zone1, zone2]
#         self.samples_per_zone = 2*[0]
#         self.comp_per_zone = 2*[0]
#         self.zone_difficulties = [200, 500]
#
# class demo():
#     def __init__(self):
#         self.tree = RegionTree(max_regions=64, n_split=10, split_min=0, alpha = 1, maxlen = 300, n_cp = 30)
#         self.tree.init_root(np.array([-1.2]), np.array([0.6]))
#         self.tree.init_grid_1D(64)
#         self.zones = zones3()
#         self.iteration = 0
#         self.dims = [0]
#         np.random.seed(None)
#
#     def iter(self):
#         goal = self.tree.sample(prop_rnd=1)
#         comp = self.zones.compute_comp(goal)
#         point = Point(goal, comp)
#         self.tree.insert_point(point)
#         self.iteration += 1
#
#     def updatefig(self, i, with_points=False):
#         for _ in range(100):
#             self.iter()
#         print(self.iteration)
#         self.tree.compute_image(dims=self.dims, with_points=with_points)
#         for line in self.tree.lines:
#             self.tree.ax.add_line(line)
#         for patch in self.tree.patches:
#             self.tree.ax.add_patch(patch)
#         if with_points:
#             x,y,z = zip(*[(point.pos[0], point.pos[1], point.val) for point in self.tree.points])
#             sizes = [0.01 + ze for ze in z]
#             self.tree.ax.scatter(x,y, s=sizes, c='red')
#         return self.tree.ax,
#
#     def run(self):
#         self.tree.displayTree(dims=self.dims)
#         ani = animation.FuncAnimation(self.tree.figure, self.updatefig, frames=100, interval=200, blit=True)
#         plt.show()
#


