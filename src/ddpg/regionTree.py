from collections import deque
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib import animation
import math


Blues = plt.get_cmap('Blues')

class Point(object):
    def __init__(self, pos, val):
        self.pos = pos
        self.val = val

class Region(Box):

    def __init__(self, low = np.array([-np.inf]), high=np.array([np.inf]), maxlen=100, n_cp = 30, shape=None):
        super(Region, self).__init__(low, high, shape)
        self.points = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.n_cp = n_cp
        self.CP = 0
        self.max_CP = 0
        self.min_CP = 0
        self.sum_CP = 0
        self.dim_split = None
        self.val_split = None

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left, maxlen=self.maxlen, n_cp = self.n_cp)
        right = Region(low_right, self.high, maxlen=self.maxlen, n_cp = self.n_cp)
        left.CP = self.CP
        right.CP = self.CP
        left.append(self.points)
        right.append(self.points)
        left.update_CP()
        right.update_CP()
        return left, right

    def append(self, points):
        for point in points:
            if self.contains(point.pos):
                self.points.append(point)
        self.update_CP()

    def update_CP(self):
        if self.size > 2*self.n_cp:
            CP = 0
            rev = list(reversed(self.points))
            for p in rev[:self.n_cp]:
                CP += p.val
            for p in rev[self.n_cp:2 * self.n_cp]:
                CP -= p.val
            self.CP = 1/2+CP / (2 * self.n_cp)
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
        return len(self.points) == self.maxlen

    # @property
    # def low(self):
    #     return self.low
    #
    # @low.setter
    # def low(self, val):
    #     self.low = val
    #
    # @property
    # def high(self):
    #     return self.high
    #
    # @high.setter
    # def high(self, val):
    #     self.high = val

class RegionTree():
    def __init__(self, n_regions, n_split, split_min):
        self.n_split = n_split
        self.split_min = split_min
        capacity = 1
        while capacity < n_regions:
            capacity *= 2
        self.capacity = capacity
        self.region_array = [Region() for _ in range(2*self.capacity)]
        self.active_regions = 0
        self.total_points = 0
        self.discarded = 0
        self.ax = None
        self.figure = None
        self.lines = []
        self.patches = []
        self.points = []

    def init_tree(self, region):
        self.region_array[1] = region
        self.update_CP_tree(1)
        self.active_regions += 1

    def insert_point(self, point):
        root = self.region_array[1]
        self._insert_point(point, root)
        self.total_points += 1

    def _insert_point(self, point, region):
        if idx >= self.capacity:
            region.append([point])
            self.update_CP_tree(idx)
        else:
            if region.full and region.is_leaf:
                self.split(idx)
            region.append([point])
            if not region.is_leaf:
                left = self.region_array[2*idx]
                right = self.region_array[2*idx + 1]
                if left.contains(point.pos):
                    self._insert_point(point, 2*idx)
                elif right.contains(point.pos):
                    self._insert_point(point, 2*idx + 1)
            else:
                self.update_CP_tree(idx)

    def update_CP_tree(self, idx):
        region = self.region_array[idx]
        region.max_CP = region.CP
        region.min_CP = region.CP
        region.sum_CP = region.CP
        idx //= 2
        while idx >= 1:
            region = self.region_array[idx]
            region_left = self.region_array[2*idx]
            region_right = self.region_array[2*idx + 1]
            split_eval = self.split_eval_1(region_left, region_right)
            to_merge = region_left.is_leaf and region_right.is_leaf and split_eval < self.split_min
            if to_merge:
                self.region_array[2 * idx + 1] = Region()
                self.region_array[2 * idx] = Region()
                print('delete split')
                region.max_CP = region.CP
                region.min_CP = region.CP
                region.sum_CP = region.CP
                region.dim_split = None
                region.val_split = None
            else:
                region.max_CP = np.max([region_left.max_CP, region_right.max_CP])
                region.min_CP = np.min([region_left.min_CP, region_right.min_CP])
                region.sum_CP = np.sum([region_left.sum_CP, region_right.sum_CP])
            idx //= 2

    def split_eval_1(self, left, right):
        return (right.CP-left.CP)**2

    def split_eval_2(self, left, right):
        return -np.abs(left.size-right.size)

    def split(self, idx):
        region = self.region_array[idx]
        eval_splits_1 = []
        eval_splits_2 = []
        for dim in range(region.shape[0]):
            for num_split, split_val in enumerate(np.linspace(region.low[dim], region.high[dim], self.n_split+2)[1:-1]):
                temp_left, temp_right = region.split(dim, split_val)
                eval_splits_1.append(self.split_eval_1(temp_left, temp_right))
                eval_splits_2.append(self.split_eval_2(temp_left, temp_right))
        eval_splits_1_norm = [(a-np.min(eval_splits_1))/(np.max(eval_splits_1)-np.min(eval_splits_1)) for a in
                              eval_splits_1]
        eval_splits_2_norm = [(a - np.min(eval_splits_2)) / (np.max(eval_splits_2) - np.min(eval_splits_2)) for
                              a in eval_splits_2]
        eval_splits = [self.lambd*x + (1-self.lambd)*y for (x,y) in zip(eval_splits_1_norm, eval_splits_2_norm)]
        best_split_idx = np.argmax(eval_splits)
        if eval_splits[best_split_idx[0]] > self.split_min:
            best_dim = best_split_idx // self.n_split
            best_val = np.linspace(region.low[best_dim], region.high[best_dim], self.n_split+2)[best_split_idx % self.n_split+1]
            best_split_left, best_split_right = region.split(best_dim, best_val)
            self.region_array[2 * idx] = best_split_left
            self.region_array[2 * idx + 1] = best_split_right
            region.dim_split = best_dim
            region.val_split = best_val
            print('splint succeeded: dim=', best_dim, ' val=', best_val)

    def displayTree(self, dims):
        self.figure = plt.figure()
        init_region = self.region_array[1]
        self.ax = plt.axes()
        self.ax.set_xlim(init_region.low[dims[0]], init_region.high[dims[0]])
        self.ax.set_ylim(init_region.low[dims[1]], init_region.high[dims[1]])

    def compute_image(self, dims, with_points=False):
        self.lines.clear()
        self.patches.clear()
        self.points.clear()
        self._compute_image(1, dims, with_points)
        print('max_cp: ', self.max_CP)
        print('min_cp', self.min_CP)

    def _compute_image(self, idx, dims, with_points=False):
        region = self.region_array[idx]
        if region.is_leaf:
            angle = (region.low[dims[0]], region.low[dims[1]])
            width = region.high[dims[0]] - region.low[dims[0]]
            height = region.high[dims[1]] - region.low[dims[1]]
            # print('region: ', region.low, ': ', region.CP)
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
            if region.dim_split == dims[0]:
                line1_xs = 2 * [region.val_split]
                line1_ys = [region.low[dims[1]], region.high[dims[1]]]
                self.lines.append(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
            elif region.dim_split == dims[1]:
                line1_ys = 2 * [region.val_split]
                line1_xs = [region.low[dims[0]], region.high[dims[0]]]
                self.lines.append(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))

            self._compute_image(2 * idx, dims)
            self._compute_image(2 * idx + 1, dims)

    def find_sum_idx(self, sum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= sum
        """
        assert 0 <= sum <= self.sum_CP + 1e-5
        idx = 1
        while not self.region_array[idx].is_leaf:
            if self.region_array[2 * idx].sum_CP > sum:
                idx = 2 * idx
            else:
                sum -= self.region_array[2 * idx].sum_CP
                idx = 2 * idx + 1
        return idx

    def sample_prop(self):
        sum = self.sum_CP
        mass = np.random.random() * sum
        idx = self.find_sum_idx(mass)
        region = self.region_array[idx]
        sample = region.sample()
        return sample

    def sample_random(self):
        region = self.region_array[1]
        sample = region.sample()
        return sample

    def sample(self, p_rnd):
        p = np.random.random()
        if p<p_rnd:
            return self.sample_random()
        else:
            return self.sample_prop()


    @property
    def max_CP(self):
        return self.region_array[1].max_CP

    @property
    def min_CP(self):
        return self.region_array[1].min_CP

    @property
    def sum_CP(self):
        return self.region_array[1].sum_CP

class demo():
    def __init__(self):
        self.tree = RegionTree(n_regions=50, n_split=10, split_min=1e-6)
        init_region = Region(np.array([-1.2, -0.07]), np.array([0.6, 0.07]), maxlen = 300, n_cp = 30)
        self.tree.init_tree(init_region)

        zone1 = Region(np.array([-1.2, -0.07]), np.array([-0.6, 0]))
        zone2 = Region(np.array([-0.6, -0.07]), np.array([0, 0]))
        zone3 = Region(np.array([0, -0.07]), np.array([0.6, 0]))
        zone4 = Region(np.array([-1.2, 0]), np.array([-0.6, 0.07]))
        zone5 = Region(np.array([-0.6, 0]), np.array([0, 0.07]))
        zone6 = Region(np.array([0, 0]), np.array([0.6, 0.07]))
        self.zones = [zone1, zone2, zone3, zone4, zone5, zone6]
        self.samples_per_zone = 6*[0]
        self.comp_per_zone = 6*[0]
        self.zone_difficulties = [0, 200, 400, 800, 1000, 1200]

        self.iteration = 0
        np.random.seed(None)

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
        # print("zone ", i-1, 'samples: ', self.samples_per_zone[i-1])
        return comp

# Kolmogorov smirnof
    def iter(self):
        goal = self.tree.sample(0.1)
        comp = self.compute_comp(goal)
        point = Point(goal, comp)
        self.tree.insert_point(point)
        self.iteration += 1

    def updatefig(self, i, with_points=False):
        self.tree.ax.lines.clear()
        self.tree.ax.patches.clear()
        for _ in range(100):
            self.iter()
        print(self.iteration)
        self.tree.compute_image(dims=[0,1], with_points=with_points)
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
        self.tree.displayTree(dims=[0,1])
        ani = animation.FuncAnimation(self.tree.figure, self.updatefig, frames=100, interval=200, blit=True)
        plt.show()

if __name__ == "__main__":
    demo = demo()
    demo.run()

