from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class Queue():
    def __init__(self, maxlen=0, n_window=0):
        self.maxlen = 2*n_window
        self.points = deque(maxlen=2*n_window)
        self.n_window = n_window
        self.CP = 0.1
        self.competence = 0.1

    def update_CP(self):
        if self.size >= 2*self.n_window:
            len = self.size
            q1 = [pt[1] for pt in list(itertools.islice(self.points, len-self.n_window, len))]
            q2 = [pt[1] for pt in list(itertools.islice(self.points, len-2*self.n_window, len-self.n_window))]
            self.CP = np.abs(np.sum(q1) - np.sum(q2)) / (2 * self.n_window)
            self.competence = np.sum(q1) / (self.n_window)

    @property
    def size(self):
        return len(self.points)

    @property
    def full(self):
        return self.size == self.maxlen


class Region(Box):

    def __init__(self, low = np.array([-np.inf]), high=np.array([np.inf]), maxlen=0, n_window=0, dims=None):
        super(Region, self).__init__(low, high)
        self.maxlen = maxlen
        self.n_window = n_window
        self.queue = Queue(maxlen, n_window)
        self.dim_split = None
        self.val_split = None
        self.dims = dims
        self.freq = 0
        self.max_competence = 0
        self.min_competence = 0
        self.max_CP = 0
        self.min_CP = 0
        self.sum_CP = 0

    def sample(self):
        return np.random.uniform(low=self.low[self.dims], high=self.high[self.dims])

    def contains(self, x):
        shape_ok = (x.shape == self.low[self.dims].shape)
        low_ok = (x >= self.low[self.dims]).all()
        high_ok = (x <= self.high[self.dims]).all()
        return shape_ok and low_ok and high_ok

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left, maxlen=self.maxlen, n_window = self.n_window, dims=self.dims)
        right = Region(low_right, self.high, maxlen=self.maxlen, n_window = self.n_window, dims=self.dims)

        left.queue.CP = self.queue.CP
        right.queue.CP = self.queue.CP
        left.queue.competence = self.queue.competence
        right.queue.competence = self.queue.competence
        for point in self.queue.points:
            if left.contains(point[0]):
                left.queue.points.append(point)
            else:
                right.queue.points.append(point)
        left.queue.update_CP()
        right.queue.update_CP()

        return left, right

    def update_CP(self):
        self.queue.update_CP()
        self.max_competence = self.queue.competence
        self.min_competence = self.queue.competence
        self.max_CP = self.queue.CP
        self.sum_CP = self.queue.CP
        self.min_CP = self.queue.CP
        assert self.queue.CP >= 0

    @property
    def is_leaf(self):
        return (self.dim_split is None)

