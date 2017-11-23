from segmentTree import SumSegmentTree, MinSegmentTree
import numpy as np
import matplotlib.pyplot as plt

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='int32'):
        self.maxlen = maxlen
        self.data = np.zeros((maxlen,) + shape).astype(dtype)
        self.next_idx = 0

    def append(self, v):
        self.data[self.next_idx] = v
        self.next_idx = (self.next_idx+1) % self.maxlen

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.maxlen:
            raise KeyError()
        return self.data[idx]


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Buffer(object):
    def __init__(self, limit, content_shape):
        self.next_idx = 0
        self.limit = limit
        self.length = 0
        self.contents = {}
        for content, shape in content_shape.items():
            self.contents[content] = RingBuffer(limit, shape=shape)

    def append(self, buffer_item):
        for name, value in self.contents.items():
            value.append(buffer_item[name])
        self.next_idx = (self.next_idx+1) % self.limit
        if self.length < self.limit:
            self.length += 1


class PrioritizedGoalBuffer(Buffer):
    def __init__(self, limit, alpha):
        self.content = {'goal': (1,)}
        self.alpha = alpha
        super(PrioritizedGoalBuffer, self).__init__(limit, self.content)

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, buffer_item, priority=None):
        """See ReplayBuffer.store_effect"""
        idx = self.next_idx
        super().append(buffer_item)
        if priority is None:
            self._it_sum[idx] = self._max_priority ** self.alpha
        else:
            self._it_sum[idx] = priority

    def sample_proportional_idx(self):
        mass = np.random.random() * self._it_sum.sum(0, self.length - 1)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self):
        # Draw such that we always have a proceeding element.
        idx = self.sample_proportional_idx()
        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value[idx])
        return idx, result

    def update_priority(self, idx, priority):
        self._it_sum[idx] = priority ** self.alpha
        self._max_priority = max(self._max_priority, priority)


def _demo():
    buffer = PrioritizedGoalBuffer(200, 1)
    samples = np.zeros((1000000), dtype=int)
    for i in range(200):
        buffer_item = {'goal': i}
        buffer.append(buffer_item, i)
    for j in range(1000000):
        idx, sample = buffer.sample()
        samples[j] = int(sample['goal'])
    bins=np.bincount(samples)
    plt.plot(range(bins.shape[0]), bins)
    plt.show()

if __name__ == "__main__":
    _demo()