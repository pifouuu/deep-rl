from collections import deque
from gym.spaces import Box
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
N=100
N_SPLIT = 10
N_CP = 30
Blues = plt.get_cmap('Blues')

class Region(Box):
    def __init__(self, low, high, shape=None):
        super(Region, self).__init__(low, high, shape)

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left)
        right = Region(low_right, self.high)
        return left, right

class Point(object):
    def __init__(self, pos, val):
        self.pos = pos
        self.val = val

class RegionNode(object):
    def __init__(self, region):
        self.points = deque(maxlen=N)
        self.CP = 0
        self.left = None
        self.right = None
        self.region = region
        self.dim_split = None
        self.val_split = None

    def split(self):
        best_split_left = None
        best_split_right = None
        best_diff = 0
        best_dim = None
        best_split_val = None
        for dim in range(self.region.shape[0]):
            for split_val in np.linspace(self.region.low[dim], self.region.high[dim], N_SPLIT+2)[1:-1]:
                temp_left, temp_right = self.region.split(dim, split_val)
                temp_left_node = RegionNode(temp_left)
                temp_right_node = RegionNode(temp_right)
                for point in self.points:
                    if temp_left_node.region.contains(point.pos):
                        temp_left_node.points.append(point)
                    elif temp_right_node.region.contains(point.pos):
                        temp_right_node.points.append(point)
                    else:
                        print("Point not in split regions")
                        raise(RuntimeError)
                temp_right_node.update_CP()
                temp_left_node.update_CP()
                diff = (temp_right_node.CP-temp_left_node.CP)**2
                if diff > best_diff:
                    best_diff = diff
                    best_split_right = temp_right_node
                    best_split_left = temp_left_node
                    best_dim = dim
                    best_split_val = split_val
        self.left = best_split_left
        self.right = best_split_right
        self.dim_split = best_dim
        self.val_split = best_split_val


    def need_split(self):
        return len(self.points) == N and self.left is None and self.right is None

    def update_CP(self):
        if self.size < 2:
            pass
        else:
            window_length = min([N_CP, self.size//2])
            self.CP = 0
            rev = reversed(self.points)
            for p in list(rev)[:window_length]:
                self.CP += p.val
            for p in list(rev)[window_length:2*window_length]:
                self.CP -= p.val
            self.CP /= 2*window_length
        return self.CP

    @property
    def size(self):
        return len(self.points)

class RegionTree(object):
    def __init__(self, region, dims):
        self.root = RegionNode(region)
        self.dims = dims
        self.CP_max = 0

    def insert_point(self, point):
        self._insert_point(point, self.root)

    def _insert_point(self, point, node):
        assert node.region.contains(point.pos)
        if node.need_split():
            node.split()
        node.points.append(point)
        if node.left is not None and node.left.region.contains(point.pos):
            self._insert_point(point, node.left)
        elif node.right is not None and node.right.region.contains(point.pos):
            self._insert_point(point, node.right)
        new_CP = node.update_CP()
        self.CP_max = max(self.CP_max, new_CP)

    def displayTree(self):
        fig, ax = plt.subplots()
        self._display_tree(self.root, ax)
        ax.set_xlim(self.root.region.low[self.dims[0]], self.root.region.high[self.dims[0]])
        ax.set_ylim(self.root.region.low[self.dims[1]], self.root.region.high[self.dims[1]])
        plt.show()

    def _display_tree(self, node, ax):
        if node.dim_split is None:
            angle = (node.region.low[self.dims[0]], node.region.low[self.dims[1]])
            width = node.region.high[self.dims[0]] - node.region.low[self.dims[0]]
            height = node.region.high[self.dims[1]] - node.region.low[self.dims[1]]
            ax.add_patch(
                patches.Rectangle(angle,
                                  width,
                                  height,
                                  fill=True,
                                  facecolor=Blues(node.CP/self.CP_max),
                                  edgecolor=None))
        else:
            if node.dim_split == self.dims[0]:
                line1_xs = 2*[node.val_split]
                line1_ys = [node.region.low[self.dims[1]], node.region.high[self.dims[1]]]
                ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))
            elif node.dim_split == self.dims[1]:
                line1_ys = 2*[node.val_split]
                line1_xs = [node.region.low[self.dims[0]], node.region.high[self.dims[0]]]
                ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=2, color='blue'))

            if node.left is not None:
                self._display_tree(node.left, ax)
            if node.right is not None:
                self._display_tree(node.right,ax)

def _demo():
    init = Region(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
    tree = RegionTree(init, [0,1])
    for _ in range(1000):
        point = Point(init.sample(), np.random.randint(10))
        tree.insert_point(point)
    tree.displayTree()

if __name__ == "__main__":
    _demo()


