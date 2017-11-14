import numpy as np
from segment_tree import SumSegmentTree, MinSegmentTree

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class ReplayBuffer(object):
    def __init__(self, limit, content_shape):
        self.contents = {}
        for content,shape in content_shape.items():
            self.contents[content] = RingBuffer(limit, shape=shape)

    def append(self, buffer_item):
        for name, value in self.contents.items():
            value.append(buffer_item[name])

class Memory():
    def __init__(self, env_wrapper, with_reward, limit):

        if with_reward:
            contents = {'state0': env_wrapper.state_shape,
                         'action': env_wrapper.action_shape,
                         'state1': env_wrapper.state_shape,
                         'reward': env_wrapper.reward_shape,
                         'terminal1':env_wrapper.terminal_shape}
        else:
            contents = {'state0': env_wrapper.state_shape,
                             'action': env_wrapper.action_shape,
                             'state1': env_wrapper.state_shape}

        self.buffer = ReplayBuffer(limit, contents)
        self.with_reward = with_reward
        self.env_wrapper = env_wrapper

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        result = {}
        for name, value in self.buffer.contents.items():
            result[name]=array_min2d(value.get_batch(batch_idxs))
        if not self.with_reward:
            result['rewards'], result['terminals1'] = \
                self.env_wrapper.evaluate_transition(result['state0'],
                                                     result['action'],
                                                     result['state1'])

        return result

    def append(self, buffer_item, training=True):
        if not training:
            return
        self.buffer.append(buffer_item)

    @property
    def nb_entries(self):
        return len(self.buffer.contents['state0'])

class HerMemory(Memory):
    def __init__(self, env_wrapper, with_reward, limit, strategy):
        """Replay buffer that does Hindsight Experience Replay
        obs_to_goal is a function that converts observations to goals
        goal_slice is a slice of indices of goal in observation
        """
        Memory.__init__(self, env_wrapper, with_reward, limit)

        self.strategy = strategy
        self.data = [] # stores current episode

    def flush(self):
        """Dump the current data into the replay buffer with (final) HER"""
        if not self.data:
            return

        state_to_goal = self.env_wrapper.state_to_goal
        state_to_obs = self.env_wrapper.state_to_obs
        obs_to_goal = self.env_wrapper.obs_to_goal

        for buffer_item in self.data:
            super().append(buffer_item)
        if self.strategy=='last':
            final_buffer = self.data[-1]
            _, reached = self.env_wrapper.evaluate_transition(final_buffer['state0'],
                                                    final_buffer['action'],
                                                    final_buffer['state1'])
            if not reached:
                final_state = self.data[-1]['state1']
                new_goal = final_state[state_to_obs][obs_to_goal]
                for buffer_item in self.data:
                    buffer_item['state0'][state_to_goal] = new_goal
                    buffer_item['state1'][state_to_goal] = new_goal
                    buffer_item['reward'], buffer_item['terminal1'] = \
                        self.env_wrapper.evaluate_transition(buffer_item['state0'],
                                                               buffer_item['action'],
                                                               buffer_item['state1'])
                    super().append(buffer_item)
        else:
            print('error her strategy')
            return
        self.data = []

    def append(self, buffer_item, training=True):
        if not training:
            return
        self.data.append(buffer_item)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = np.random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
