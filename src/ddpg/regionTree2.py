from collections import deque
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib import animation
import itertools

from ddpg.memory import SARSTMemory, EpisodicHerSARSTMemory


Blues = plt.get_cmap('Blues')

class Point(object):
    def __init__(self, state, val):
        self.state = state
        self.val = val

class Region(Box):

    def __init__(self, low = np.array([-np.inf]), high=np.array([np.inf]), maxlen=0, n_cp=0):
        super(Region, self).__init__(low, high)
        self.maxlen = maxlen
        self.points = deque(maxlen=self.maxlen)
        self.n_cp = n_cp
        self.CP = 0
        self.max_CP = 0
        self.min_CP = 0
        self.sum_CP = 0
        self.dim_split = None
        self.val_split = None

    def sample(self, dims=None):
        if dims is not None:
            return np.random.uniform(low=self.low[dims], high=self.high[dims], size=self.low[dims].shape)
        else:
            return super(Region, self).sample()

    def contains(self, point, dims=None):
        x = point.state
        if dims is not None:
            return x.shape == self.shape and (x >= self.low[dims]).all() and (x <= self.high[dims]).all()
        else:
            return super(Region,self).contains(x)

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left, maxlen=self.maxlen, n_cp = self.n_cp)
        right = Region(low_right, self.high, maxlen=self.maxlen, n_cp = self.n_cp)
        left.CP = self.CP
        right.CP = self.CP
        left_points = []
        right_points = []
        for point in self.points:
            if left.contains(point):
                left_points.append(point)
            else:
                right_points.append(point)
        left.add(left_points)
        right.add(right_points)
        left.update_CP()
        right.update_CP()
        return left, right

    def add(self, points):
        for point in points:
            self.points.append(point)
        self.update_CP()

    def update_CP(self):
        if self.size > 2*self.n_cp:
            len = self.size
            q1 = [pt.val for pt in list(itertools.islice(self.points, len-self.n_cp, len))]
            q2 = [pt.val for pt in list(itertools.islice(self.points, len-2*self.n_cp, len-self.n_cp))]
            self.CP = 1/2 + (np.sum(q1)-np.sum(q2))/(2*self.n_cp)
        self.max_CP = self.CP
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
    def __init__(self, space, dims, buffer, max_regions, n_split, split_min, alpha, maxlen, n_cp):
        self.n_split = n_split
        self.split_min = split_min
        self.maxlen = maxlen
        self.n_cp = n_cp
        self.max_regions = max_regions
        self.alpha = alpha
        self.dims = dims
        self.figure_dims = dims

        self.buffer = buffer

        self.ax = None
        self.figure = None
        self.lines = []
        self.patches = []
        self.points = []

        capacity = 1
        while capacity < max_regions:
            capacity *= 2
        self.capacity = capacity
        self.region_array = [Region() for _ in range(2 * self.capacity)]
        self.region_array[1] = Region(space.low, space.high, maxlen=self.maxlen, n_cp=self.n_cp)
        self.update_CP_tree(1)
        self.n_leaves = 1

    def end_episode(self, goal_reached):
        self.buffer.end_episode(goal_reached)

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def append(self, buffer_item):
        self.buffer.append(buffer_item)

    def build_exp(self, state, action, next_state, reward, terminal):
        return self.buffer.build_exp(state, action, next_state, reward, terminal)

    def update(self, indices, vals):
        stg = self.buffer.env.state_to_goal
        minval = self.buffer.env.reward_range[0] / (1 - 0.99)
        maxval = self.buffer.env.reward_range[1]
        states0 = [self.buffer.contents['state0'][idx][stg] for idx in indices]
        corr_vals = [(val-minval)/(maxval-minval) for val in vals]
        self.insert([Point(state,val) for state,val in zip(states0,corr_vals)])

    def init_grid_1D(self, n):
        assert n & (n-1) == 0 #n must be a power of 2
        assert len(self.dims) == 1
        self._init_grid_1D(1, n)

    def _init_grid_1D(self, idx ,n):
        if n > 1:
            region = self.region_array[idx]
            low = region.low[self.dims[0]]
            high = region.high[self.dims[0]]
            val_split = (high+low)/2
            self.region_array[2 * idx], self.region_array[2 * idx + 1] = region.split(self.dims[0], val_split)
            region.dim_split = self.dims[0]
            region.val_split = val_split
            self.n_leaves += 1
            self._init_grid_1D(2 * idx, n/2)
            self._init_grid_1D(2 * idx + 1, n/2)

    def insert(self, points):
        self._insert(points, 1)

    def _insert(self, points, idx):
        region = self.region_array[idx]
        region.add(points)

        if not region.is_leaf:
            left = self.region_array[2 * idx]
            left_points = []
            right_points = []
            for point in points:
                if left.contains(point):
                    left_points.append(point)
                else:
                    right_points.append(point)
            self._insert(left_points, 2 * idx)
            self._insert(right_points, 2 * idx + 1)
        else:
            if region.full and idx < self.capacity:
                self.split(idx)
            self.update_CP_tree(idx)


    def update_CP_tree(self, idx):
        region = self.region_array[idx]
        region.max_CP = region.CP
        region.min_CP = region.CP
        region.sum_CP = region.CP
        idx //= 2
        while idx >= 1:
            region = self.region_array[idx]
            left = self.region_array[2*idx]
            right = self.region_array[2*idx + 1]
            split_eval = self.split_eval_1(left, right)
            to_merge = left.is_leaf and right.is_leaf and split_eval < self.split_min
            if to_merge:
                region.max_CP = region.CP
                region.min_CP = region.CP
                region.sum_CP = region.CP
                region.dim_split = None
                region.val_split = None
                self.region_array[2 * idx] = None
                self.region_array[2 * idx + 1] = None
                self.n_leaves -= 1
                print('merge')
            else:
                region.max_CP = np.max([left.max_CP, right.max_CP])
                region.min_CP = np.min([left.min_CP, right.min_CP])
                region.sum_CP = np.sum([left.sum_CP, right.sum_CP])
            idx //= 2

    def split_eval_1(self, left, right):
        return (right.CP-left.CP)**2

    def split_eval_2(self, left, right):
        return -np.abs(left.size-right.size)

    def split(self, idx):
        eval_splits_1 = []
        eval_splits_2 = []
        if self.n_leaves < self.max_regions:
            region = self.region_array[idx]
            for dim in self.dims:
                for num_split, split_val in enumerate(np.linspace(region.low[dim], region.high[dim], self.n_split+2)[1:-1]):
                    temp_left, temp_right = region.split(dim, split_val)
                    eval_splits_1.append(self.split_eval_1(temp_left, temp_right))
                    eval_splits_2.append(self.split_eval_2(temp_left, temp_right))
            width1 = np.max(eval_splits_1)-np.min(eval_splits_1)
            if width1 != 0:
                eval_splits_1_norm = [(a-np.min(eval_splits_1)) / width1 for a in eval_splits_1]
                width2 = np.max(eval_splits_2) - np.min(eval_splits_2)
                eval_splits_2_norm = [(a - np.min(eval_splits_2)) / width2 for a in eval_splits_2]
                eval_splits = [self.alpha*x + (1-self.alpha)*y for (x,y) in zip(eval_splits_1_norm, eval_splits_2_norm)]
                split_idx = np.argmax(eval_splits)
                if eval_splits_1[split_idx] > self.split_min:
                    region.dim_split = self.dims[split_idx // self.n_split]
                    region.val_split = np.linspace(region.low[region.dim_split], region.high[region.dim_split], self.n_split+2)[split_idx % self.n_split+1]
                    self.region_array[2 * idx], self.region_array[2 * idx + 1] = region.split(region.dim_split, region.val_split)
                    print('splint succeeded: dim=', region.dim_split, ' val=', region.val_split)
                    self.n_leaves += 1

    def init_display(self, figure_dims=None):
        self.figure_dims = figure_dims
        self.figure = plt.figure()
        self.ax = plt.axes()
        self.ax.set_xlim(self.root.low[self.figure_dims[0]], self.root.high[self.figure_dims[0]])
        if len(self.figure_dims)>1:
            self.ax.set_ylim(self.root.low[self.figure_dims[1]], self.root.high[self.figure_dims[1]])
        else:
            self.ax.set_ylim(0, 1)
        plt.ion()
        plt.show()


    def compute_image(self, with_points=False):
        self.lines.clear()
        self.patches.clear()
        self.points.clear()
        self._compute_image(1, with_points)

    def plot_image(self, with_points=False):
        self.ax.lines.clear()
        self.ax.patches.clear()
        for line in self.lines:
            self.ax.add_line(line)
        for patch in self.patches:
            self.ax.add_patch(patch)
        if with_points:
            x, y, z = zip(*[(point.pos[0], point.pos[1], point.val) for point in self.points])
            sizes = [0.01 + ze for ze in z]
            self.ax.scatter(x, y, s=sizes, c='red')
        plt.draw()
        plt.pause(0.001)

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
            if self.max_CP == 0:
                color = 0
            else:
                color = region.CP/self.max_CP
            self.patches.append(patches.Rectangle(angle,
                                  width,
                                  height,
                                  fill=True,
                                  facecolor=Blues(color),
                                  edgecolor=None,
                                  alpha=0.8))
            if with_points:
                for point in region.points:
                    self.points.append(point)
        else:
            if region.dim_split == self.figure_dims[0]:
                line1_xs = 2 * [region.val_split]
                line1_ys = [low1, high1]
                self.lines.append(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
            elif len(self.figure_dims)>1 and region.dim_split == self.figure_dims[1]:
                line1_ys = 2 * [region.val_split]
                line1_xs = [region.low[self.figure_dims[0]], region.high[self.figure_dims[0]]]
                self.lines.append(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))

            self._compute_image(2 * idx, self.figure_dims)
            self._compute_image(2 * idx + 1, self.figure_dims)

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
        sample = region.sample(self.dims)
        return sample

    def sample_random(self):
        sample = self.root.sample(self.dims)
        return sample

    def sample_goal(self, prop_rnd=1):
        p = np.random.random()
        if p<prop_rnd:
            return self.sample_random()
        else:
            return self.sample_prop()

    @property
    def root(self):
        return self.region_array[1]

    @property
    def max_CP(self):
        return self.root.max_CP

    @property
    def min_CP(self):
        return self.root.min_CP

    @property
    def sum_CP(self):
        return self.root.sum_CP

class zones():
    def __init__(self):

        self.zones = []
        self.samples_per_zone = []
        self.comp_per_zone = []
        self.zone_difficulties = []

    def compute_comp(self, goal):
        goal_zone = None
        i = 0
        while goal_zone is None:
            if self.zones[i].contains(goal):
                goal_zone = self.zones[i]
            i += 1
        n_samples = self.samples_per_zone[i - 1]

        if n_samples < self.zone_difficulties[i - 1]:
            comp = 0
        else:
            comp = np.min([1, (n_samples - self.zone_difficulties[i - 1]) / 1000])

        self.samples_per_zone[i - 1] += 1
        return comp

class zones1(zones):
    def __init__(self):
        super(zones1, self).__init__()
        zone1 = Region(np.array([-1.2, -0.07]), np.array([-0.6, 0]))
        zone2 = Region(np.array([-0.6, -0.07]), np.array([0, 0]))
        zone3 = Region(np.array([0, -0.07]), np.array([0.6, 0]))
        zone4 = Region(np.array([-1.2, 0]), np.array([-0.6, 0.07]))
        zone5 = Region(np.array([-0.6, 0]), np.array([0, 0.07]))
        zone6 = Region(np.array([0, 0]), np.array([0.6, 0.07]))
        self.zones = [zone1, zone2, zone3, zone4, zone5, zone6]
        self.samples_per_zone = 6 * [0]
        self.comp_per_zone = 6 * [0]
        self.zone_difficulties = [0, 200, 400, 800, 1000, 1200]

class zones2(zones):
    def __init__(self):
        super(zones2, self).__init__()
        zone1 = Region(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
        self.zones = [zone1]
        self.samples_per_zone = [0]
        self.comp_per_zone = [0]
        self.zone_difficulties = [200]

class zones3(zones):
    def __init__(self):
        super(zones3, self).__init__()
        zone1 = Region(np.array([-1.2]), np.array([0]))
        zone2 = Region(np.array([0]), np.array([0.6]))
        self.zones = [zone1, zone2]
        self.samples_per_zone = 2*[0]
        self.comp_per_zone = 2*[0]
        self.zone_difficulties = [200, 500]

class demo():
    def __init__(self):
        self.tree = RegionTree(max_regions=64, n_split=10, split_min=0, alpha = 1, maxlen = 300, n_cp = 30)
        self.tree.init_root(np.array([-1.2]), np.array([0.6]))
        self.tree.init_grid_1D(64)
        self.zones = zones3()
        self.iteration = 0
        self.dims = [0]
        np.random.seed(None)

    def iter(self):
        goal = self.tree.sample(prop_rnd=1)
        comp = self.zones.compute_comp(goal)
        point = Point(goal, comp)
        self.tree.insert_point(point)
        self.iteration += 1

    def updatefig(self, i, with_points=False):
        for _ in range(100):
            self.iter()
        print(self.iteration)
        self.tree.compute_image(dims=self.dims, with_points=with_points)
        for line in self.tree.lines:
            self.tree.ax.add_line(line)
        for patch in self.tree.patches:
            self.tree.ax.add_patch(patch)
        if with_points:
            x,y,z = zip(*[(point.pos[0], point.pos[1], point.val) for point in self.tree.points])
            sizes = [0.01 + ze for ze in z]
            self.tree.ax.scatter(x,y, s=sizes, c='red')
        return self.tree.ax,

    def run(self):
        self.tree.displayTree(dims=self.dims)
        ani = animation.FuncAnimation(self.tree.figure, self.updatefig, frames=100, interval=200, blit=True)
        plt.show()

if __name__ == "__main__":
    demo = demo()
    demo.run()

