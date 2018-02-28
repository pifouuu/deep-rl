from collections import deque
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib import animation
import itertools


Blues = plt.get_cmap('Blues')

class Point(object):
    def __init__(self, pos, val):
        self.pos = pos
        self.val = val

class Region(Box):

    def __init__(self, low, high, maxlen=0, n_cp=0):
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
        self.left = None
        self.right = None
        self.mother = None

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
        left.mother = self
        right.mother = self
        self.right = right
        self.left = left
        return left, right

    def append(self, points):
        for point in reversed(points):
            if self.contains(point.pos):
                self.points.appendleft(point)
        self.update_CP()

    def update_CP(self):
        if self.size > 2*self.n_cp:
            q1 = [pt.val for pt in list(itertools.islice(self.points, 0, self.n_cp))]
            q2 = [pt.val for pt in list(itertools.islice(self.points, self.n_cp, 2*self.n_cp))]
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

class RegionTree():
    def __init__(self, max_regions, n_split, split_min, lambd, maxlen, n_cp):
        self.n_split = n_split
        self.split_min = split_min
        self.maxlen = maxlen
        self.n_cp = n_cp
        self.max_regions = max_regions
        self.total_points = 0
        self.ax = None
        self.figure = None
        self.lines = []
        self.patches = []
        self.points = []
        self.root = None
        self.lambd = lambd
        self.n_leaves = 0

    def init_tree(self, low, high):
        self.root = Region(low, high, maxlen=self.maxlen, n_cp = self.n_cp)
        self.update_CP_tree(self.root)
        self.n_leaves += 1

    def insert_point(self, point):
        self._insert_point(point, self.root)
        self.total_points += 1

    def _insert_point(self, point, region):
        region.append([point])
        if not region.is_leaf:
            if region.left.contains(point.pos):
                self._insert_point(point, region.left)
            elif region.right.contains(point.pos):
                self._insert_point(point, region.right)
        else:
            if region.full:
                self.split(region)
            self.update_CP_tree(region)

    def update_CP_tree(self, region):
        region.max_CP = region.CP
        region.min_CP = region.CP
        region.sum_CP = region.CP
        region = region.mother
        while region is not None:
            left = region.left
            right = region.right
            split_eval = self.split_eval_1(left, right)
            to_merge = left.is_leaf and right.is_leaf and split_eval < self.split_min
            if to_merge:
                region.max_CP = region.CP
                region.min_CP = region.CP
                region.sum_CP = region.CP
                region.dim_split = None
                region.val_split = None
                self.n_leaves -= 1
                print('merge')
            else:
                region.max_CP = np.max([left.max_CP, right.max_CP])
                region.min_CP = np.min([left.min_CP, right.min_CP])
                region.sum_CP = np.sum([left.sum_CP, right.sum_CP])
            region = region.mother

    def split_eval_1(self, left, right):
        return (right.CP-left.CP)**2

    def split_eval_2(self, left, right):
        return -np.abs(left.size-right.size)

    def split(self, region):
        eval_splits_1 = []
        eval_splits_2 = []
        if self.n_leaves < self.max_regions:
            for dim in range(region.shape[0]):
                for num_split, split_val in enumerate(np.linspace(region.low[dim], region.high[dim], self.n_split+2)[1:-1]):
                    temp_left, temp_right = region.split(dim, split_val)
                    eval_splits_1.append(self.split_eval_1(temp_left, temp_right))
                    eval_splits_2.append(self.split_eval_2(temp_left, temp_right))
            width1 = np.max(eval_splits_1)-np.min(eval_splits_1)
            if width1 != 0:
                eval_splits_1_norm = [(a-np.min(eval_splits_1)) / width1 for a in eval_splits_1]
                width2 = np.max(eval_splits_2) - np.min(eval_splits_2)
                eval_splits_2_norm = [(a - np.min(eval_splits_2)) / width2 for a in eval_splits_2]
                eval_splits = [self.lambd*x + (1-self.lambd)*y for (x,y) in zip(eval_splits_1_norm, eval_splits_2_norm)]
                idx = np.argmax(eval_splits)
                if eval_splits_1[idx] > self.split_min:
                    region.dim_split = idx // self.n_split
                    region.val_split = np.linspace(region.low[region.dim_split], region.high[region.dim_split], self.n_split+2)[idx % self.n_split+1]
                    region.left, region.right = region.split(region.dim_split, region.val_split)
                    print('splint succeeded: dim=', region.dim_split, ' val=', region.val_split)
                    self.n_leaves += 1

    def displayTree(self, dims):
        self.figure = plt.figure()
        self.ax = plt.axes()
        self.ax.set_xlim(self.root.low[dims[0]], self.root.high[dims[0]])
        if len(dims)>1:
            self.ax.set_ylim(self.root.low[dims[1]], self.root.high[dims[1]])
        else:
            self.ax.set_ylim(0, 1)

    def compute_image(self, dims, with_points=False):
        self.lines.clear()
        self.patches.clear()
        self.points.clear()
        self._compute_image(self.root, dims, with_points)
        print('max_cp: ', self.max_CP)
        print('min_cp', self.min_CP)

    def _compute_image(self, region, dims, with_points=False):

        if len(dims) > 1:
            low1 = region.low[dims[1]]
            high1 = region.high[dims[1]]
        else:
            low1 = 0
            high1 = 1

        if region.is_leaf:
            angle = (region.low[dims[0]], low1)
            width = region.high[dims[0]] - region.low[dims[0]]
            height = high1 - low1

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
                line1_ys = [low1, high1]
                self.lines.append(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
            elif len(dims)>1 and region.dim_split == dims[1]:
                line1_ys = 2 * [region.val_split]
                line1_xs = [region.low[dims[0]], region.high[dims[0]]]
                self.lines.append(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))

            self._compute_image(region.left, dims)
            self._compute_image(region.right, dims)

    def find_prop_region(self, sum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= sum
        """
        assert 0 <= sum <= self.sum_CP + 1e-5
        region = self.root
        while not region.is_leaf:
            s = region.left.sum_CP
            if s > sum:
                region = region.left
            else:
                sum -= s
                region = region.right
        return region

    def sample_prop(self):
        sum = self.sum_CP
        mass = np.random.random() * sum
        region = self.find_prop_region(mass)
        sample = region.sample()
        return sample

    def sample_random(self):
        sample = self.root.sample()
        return sample

    def sample(self, prop_rnd=1):
        p = np.random.random()
        if p<prop_rnd:
            return self.sample_random()
        else:
            return self.sample_prop()


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
        self.tree = RegionTree(max_regions=40, n_split=10, split_min=1e-8, lambd = 1, maxlen = 300, n_cp = 30)
        self.tree.init_tree(np.array([-1.2]), np.array([0.6]))
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
        self.tree.ax.lines.clear()
        self.tree.ax.patches.clear()
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

