import numpy as np
from ddpg.regions import Region

class FixedRegionsMemory():
    def __init__(self, space, dims, buffer, actor, critic, N, n_split, split_min, alpha, maxlen, n_window, render, sampler):
        self.n_split = n_split
        self.split_min = split_min
        self.maxlen = maxlen
        self.n_window = n_window
        self.alpha = alpha
        self.dims = dims
        self.figure_dims = dims

        self.buffer = buffer
        self.sampler = sampler

        self.ax = None
        self.figure = None
        self.lines = []
        self.patches = []
        self.n_regions = N

        capacity = 1
        while capacity < self.n_regions:
            capacity *= 2
        self.capacity = capacity
        self.region_array = [Region() for _ in range(2 * self.capacity)]
        self.region_array[1] = Region(space.low, space.high, maxlen=self.maxlen, n_window=self.n_window, dims=dims)
        self.update_CP_tree()
        self.n_leaves = 1
        self.actor = actor
        self.critic = critic

        self.initialize()


    def end_episode(self, goal_reached):
        self.buffer.end_episode(goal_reached)
        if self.buffer.env.goal_parameterized:
            self.update_competence()
            self.update_tree()
            self.update_CP_tree()
            self.compute_image()

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def append(self, buffer_item):
        self.buffer.append(buffer_item)

    def build_exp(self, state, action, next_state, reward, terminal):
        return self.buffer.build_exp(state, action, next_state, reward, terminal)

    def eval_goal(self, goal):
        start = self.buffer.env.start
        state = np.array([np.hstack([start, goal])])
        action = self.actor.predict(state)
        q_value = self.critic.predict(state, action)
        q_value = np.squeeze(q_value)
        positive_q_value = q_value + 50
        return positive_q_value

    def update_competence(self):
        dims = self.buffer.env.internal
        for idx in range(self.capacity):
            region = self.region_array[idx + self.capacity]
            goal = np.random.uniform(region.low[dims], region.high[dims])
            competence = self.eval_goal(goal)
            self.insert((goal, competence))

    def find_prop_region(self, mass):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= sum
        """
        assert 0 <= mass <= self.sum_CP + 1e-5
        idx = 1
        while not self.region_array[idx].is_leaf:
            s = self.region_array[2 * idx].sum_CP
            if s > mass:
                idx = 2 * idx
            else:
                mass -= s
                idx = 2 * idx + 1
        return self.region_array[idx]

    def sample_goal(self):
        if self.sampler == 'prio' and np.random.random() > self.alpha:
            sum = self.sum_CP
            mass = np.random.random() * sum
            region = self.find_prop_region(mass)
        else:
            idx = np.random.randint(self.capacity)
            region = self.region_array[idx+self.capacity]

        region.freq += 1
        dims = self.buffer.env.internal
        goal = np.random.uniform(region.low[dims], region.high[dims])

        return goal

    def insert(self, point):
        self._insert(point, 1)

    def _insert(self, point, idx):
        region = self.region_array[idx]
        region.queue.points.append(point)
        if not region.is_leaf:
            left = self.region_array[2 * idx]
            if left.contains(point[0]):
                self._insert(point, 2 * idx)
            else:
                self._insert(point, 2 * idx + 1)

    def initialize(self):
        assert self.n_regions & (self.n_regions-1) == 0 #n must be a power of 2
        self._divide(1, self.n_regions, 0)

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
        region.queue.update_CP()
        if not region.is_leaf:
            self._update_tree(2 * idx)
            self._update_tree(2 * idx + 1)

    def update_CP_tree(self):
        self._update_CP_tree(1)

    def _update_CP_tree(self, idx):
        region = self.region_array[idx]
        if region.is_leaf:
            region.max_CP = region.queue.CP
            region.min_CP = region.queue.CP
            region.sum_CP = region.queue.CP
            region.max_competence = region.queue.competence
            region.min_competence = region.queue.competence
            region.sum_competence = region.queue.competence

        else:
            left = self.region_array[2 * idx]
            right = self.region_array[2 * idx + 1]
            self._update_CP_tree(2 * idx)
            self._update_CP_tree(2 * idx + 1)
            region.max_CP = np.max([left.max_CP, right.max_CP])
            region.min_CP = np.min([left.min_CP, right.min_CP])
            region.sum_CP = np.sum([left.sum_CP, right.sum_CP])
            region.max_competence = np.max([left.max_competence, right.max_competence])
            region.min_competence = np.min([left.min_competence, right.min_competence])

    def compute_image(self):
        self.lines.clear()
        self.patches.clear()
        self._compute_image(1)

    def _compute_image(self, idx):
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
            self.patches.append({'angle': angle,
                                 'width': width,
                                 'height': height,
                                 'cp': region.queue.CP,
                                 'competence': region.queue.competence,
                                 'freq': region.freq
                                 })
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

            self._compute_image(2 * idx)
            self._compute_image(2 * idx + 1)

    def stats(self):
        stats = {}
        stats['max_CP'] = self.max_CP
        stats['min_CP'] = self.min_CP
        stats['max_comp'] = self.max_competence
        stats['min_comp'] = self.min_competence
        stats['patches'] = self.patches
        stats['lines'] = self.lines
        return stats

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
    def min_CP(self):
        return self.root.min_CP

    @property
    def sum_CP(self):
        return self.root.sum_CP




