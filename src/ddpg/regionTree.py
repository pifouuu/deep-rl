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
        self.is_defined = False

    def full(self):
        return len(self.points) == self.maxlen

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left, maxlen=self.maxlen, n_cp = self.n_cp)
        right = Region(low_right, self.high, maxlen=self.maxlen, n_cp = self.n_cp)
        return left, right

    def append(self, point):
        self.points.append(point)
        self.update_CP()

    def update_CP(self):
        CP = 0
        if self.size < 2:
            pass
        else:
            window_length = min([self.n_cp, self.size // 2])
            rev = reversed(self.points)
            for p in list(rev)[:window_length]:
                CP += p.val
            for p in list(rev)[window_length:2 * window_length]:
                CP -= p.val
            CP /= 2 * window_length
        self.CP = CP

    @property
    def size(self):
        return len(self.points)

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

    def init_tree(self, region):
        self.region_array[1] = region
        self.region_array[1].is_defined = True
        self.update_CP_tree(1)
        self.active_regions += 1

    def insert_point(self, point):
        self._insert_point(point, 1)
        self.total_points += 1
        # assert self.total_points == self.points_in_leaves()

    def _insert_point(self, point, idx):
        region = self.region_array[idx]
        if not region.contains(point.pos):
            print('point ', point.pos[0], point.pos[1], ' not in region')
            pass
        if idx >= self.capacity:
            region.append(point)
            self.update_CP_tree(idx)
            self.discarded += 1
        else:
            if region.full() and self.is_leaf(idx):
                self.split(idx)
            region.append(point)
            if not self.is_leaf(idx):
                left = self.region_array[2*idx]
                right = self.region_array[2*idx + 1]
                if left.contains(point.pos):
                    self._insert_point(point, 2*idx)
                elif right.contains(point.pos):
                    self._insert_point(point, 2*idx + 1)
            else:
                self.update_CP_tree(idx)

    def is_leaf(self, idx):
        try:
            left = self.region_array[2 * idx].is_defined
            right = self.region_array[2 * idx + 1].is_defined
        except IndexError:
            return True
        return not (left or right)

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
            if self.split_eval_1(region_left, region_right) < self.split_min:
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
        best_split_left = None
        best_split_right = None
        best_split_eval = -np.inf
        best_dim = None
        best_val = None
        split = False
        region = self.region_array[idx]
        for dim in range(region.shape[0]):
            for split_val in np.linspace(region.low[dim], region.high[dim], self.n_split+2)[1:-1]:
                temp_left, temp_right = region.split(dim, split_val)
                for point in region.points:
                    if temp_left.contains(point.pos):
                        temp_left.points.append(point)
                    elif temp_right.contains(point.pos):
                        temp_right.points.append(point)
                    else:
                        print("Point not in split regions")
                        raise(RuntimeError)
                temp_left.update_CP()
                temp_right.update_CP()
                split_eval = self.split_eval_1(temp_left, temp_right)
                if split_eval > self.split_min and split_eval > best_split_eval:
                    best_split_eval = split_eval
                    best_split_right = temp_right
                    best_split_left = temp_left
                    best_dim = dim
                    best_val = split_val
                    split = True
        if split:
            best_split_left.is_defined = True
            best_split_left.max_CP = best_split_left.CP
            best_split_left.sum_CP = best_split_left.CP
            best_split_left.min_CP = best_split_left.CP
            self.region_array[2 * idx] = best_split_left

            best_split_right.is_defined = True
            best_split_right.max_CP = best_split_right.CP
            best_split_right.sum_CP = best_split_right.CP
            best_split_right.min_CP = best_split_right.CP
            self.region_array[2 * idx + 1] = best_split_right

            self.active_regions += 2

            region.dim_split = best_dim
            region.val_split = best_val
        else:
            print('no split')

    def points_in_leaves(self):
        return self._point_in_leaves(1)

    def _point_in_leaves(self, idx):
        res = 0
        if idx >= self.capacity or self.is_leaf(idx) :
            res += self.region_array[idx].size
        else:
            res += self._point_in_leaves(2*idx)
            res += self._point_in_leaves(2*idx + 1)
        return res

    def displayTree(self, dims):
        self.fig = plt.figure()
        init_region = self.region_array[1]
        self.ax = plt.axes()
        self.ax.set_xlim(init_region.low[dims[0]], init_region.high[dims[0]])
        self.ax.set_ylim(init_region.low[dims[1]], init_region.high[dims[1]])

        point_list = self._display_tree(1, dims)
        # x,y = zip(*[(point[0],point[1]) for point in point_list])
        # print(len(x))
        # ax.scatter(x,y, s=0.2, c='red')
        # plt.show()

    def _display_tree(self, idx, dims):
        region = self.region_array[idx]
        point_list = []
        point_list_left = []
        point_list_right = []
        if region.dim_split is None:
            angle = (region.low[dims[0]], region.low[dims[1]])
            width = region.high[dims[0]] - region.low[dims[0]]
            height = region.high[dims[1]] - region.low[dims[1]]
            self.ax.add_patch(
                patches.Rectangle(angle,
                                  width,
                                  height,
                                  fill=True,
                                  facecolor=Blues(region.CP/self.max_CP),
                                  edgecolor=None,
                                  alpha=1))
            for points in region.points:
                point_list.append(points.pos)
        else:
            if region.dim_split == dims[0]:
                line1_xs = 2*[region.val_split]
                line1_ys = [region.low[dims[1]], region.high[dims[1]]]
                self.ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
            elif region.dim_split == dims[1]:
                line1_ys = 2*[region.val_split]
                line1_xs = [region.low[dims[0]], region.high[dims[0]]]
                self.ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))

            point_list_left += self._display_tree(2*idx, dims)
            point_list_right += self._display_tree(2*idx + 1, dims)

        return point_list+point_list_left+point_list_right

    def find_sum_idx(self, sum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= sum
        """
        assert 0 <= sum <= self.sum_CP + 1e-5
        idx = 1
        while not self.is_leaf(idx):
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

    def sample(self):
        p = np.random.random()
        if p<0.3:
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

def sigmoid(x, l):
    return 1 / (1 + math.exp(-l*x))

class demo():
    def __init__(self):
        self.tree = RegionTree(10, 10, 10)
        init_region = Region(np.array([-1.2, -0.07]), np.array([0.6, 0.07]), maxlen = 1000, n_cp = 10)
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
        self.zone_difficulties = [1, 2, 4, 7, 11, 16]

        self.iteration = 0

    def iter(self,iteration):
        goal = self.tree.sample()

        goal_zone = None
        i = 0
        while goal_zone is None:
            if self.zones[i].contains(goal):
                goal_zone = self.zones[i]
            i += 1
        n_samples = self.samples_per_zone[i-1]
        # comp = sigmoid(n_samples-self.zone_difficulties[i-1]*10, l=2)
        if i<4:
            if iteration<4000:
                comp = n_samples
            else:
                comp = 0
        else:
            if iteration<4000:
                comp = 0
            else:
                comp = n_samples
        self.comp_per_zone[i-1] += comp
        # comp = np.min([1, (n_samples / 500) ** (self.zone_difficulties[i - 1] / 6)])
        # print('goal zone: ', i-1, ', comp: ', comp)

        self.samples_per_zone[i - 1] += 1
        point = Point(goal, comp)
        self.tree.insert_point(point)
        self.iteration += 1

    def run(self):
        figs = []
        for i in range(10000):
            self.iter(i)
            if i % 100 == 0 and i > 0:
                print('iter ', i)
                for zone, (nb, comp) in enumerate(zip(self.samples_per_zone,self.comp_per_zone)):
                    print('zone ', zone, ', nb: ', nb, ', comp: ', comp/nb)
                print('\n')
                # self.samples_per_zone = 6*[0]
                # self.comp_per_zone = 6*[0]
                self.tree.displayTree(dims=[0, 1])
                self.tree.fig.canvas.draw()
                figs.append(np.array(self.tree.fig.canvas.renderer._renderer))
                plt.close(self.tree.fig)
        print('discarded: ', self.tree.discarded)
        return figs


if __name__ == "__main__":
    demo = demo()
    figs = demo.run()

    fig = plt.figure()
    im = plt.imshow(figs[0], animated=True)


    def updatefig(i):
        im.set_array(figs[i + 1])
        return im,


    ani = animation.FuncAnimation(fig, updatefig, frames=98, interval=50)
    plt.show()


