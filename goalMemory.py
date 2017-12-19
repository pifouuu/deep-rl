import numpy as np
from segmentTree import SumSegmentTree, MinSegmentTree

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
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

    def __setitem__(self, idx, val):
        if idx < 0 or idx >= self.maxlen:
            raise KeyError()
        self.data[idx] = val


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class ContentBuffer():
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

class PrioritizedBuffer(ContentBuffer):
    def __init__(self, limit, alpha, content):
        super(PrioritizedBuffer, self).__init__(limit, content)
        self.alpha = alpha

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._max_priority = 1e5
        self._min_priority = 1e-5

    def append(self, buffer_item, priority=None):
        """See ReplayBuffer.store_effect"""
        idx = self.next_idx
        super().append(buffer_item)
        if priority is None:
            self._it_sum[idx] = self._min_priority
        else:
            self._it_sum[idx] = np.clip(priority ** self.alpha, self._min_priority, self._max_priority)

    def sample_proportional_idx(self):
        sum = self._it_sum.sum()
        mass = np.random.random() * sum
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample_prioritized(self):
        # Draw such that we always have a proceeding element.
        idx = self.sample_proportional_idx()
        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value[idx])
        return idx, result

    def update_priority(self, idx, priority):
        self._it_sum[idx] = np.clip(priority ** self.alpha, self._min_priority, self._max_priority)

    def end_episode(self):
        pass

class PrioritizedMemory(PrioritizedBuffer):
    def __init__(self, contents_shape, limit, alpha):
        super(PrioritizedMemory, self).__init__(limit, alpha, contents_shape)

    def end_episode(self):
        pass

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value.get_batch(batch_idxs))
        return batch_idxs, result

    def sample_prioritized(self):
        return super(PrioritizedMemory, self).sample_prioritized()

    @property
    def nb_entries(self):
        return len(self.contents['state0'])

class GSARSTMemory(PrioritizedMemory):
    def __init__(self, env_wrapper, limit, alpha, actor, critic):
        self.contents_shape = {'state0': env_wrapper.state_shape,
                        'action': env_wrapper.action_shape,
                        'state1': env_wrapper.state_shape,
                        'reward': env_wrapper.reward_shape,
                         'terminal': env_wrapper.terminal_shape,
                         'state1_target_val': (1,),
                         'state0_val': (1,),
                         'last_seen': (1,)}

        self.critic = critic
        self.actor = actor

        super(GSARSTMemory, self).__init__(self.contents_shape, limit, alpha)

    def build_exp(self, state, action, next_state, reward, terminal):
        dict = {'state0': state,
                'action': action,
                'state1': next_state,
                'reward': reward,
                'terminal': terminal,
                'state1_target_val': self.critic.predict_target(next_state,
                                                                self.actor.predict_target(next_state)),
                'state0_val': self.critic.predict(state, action),
                'last_seen': 0}
        return dict

    def sample_goal(self):
        idx, dict = super(GSARSTMemory, self).sample_prioritized()
        return np.reshape(dict['state0'], (1,))

    def update_priorities(self, batch_idxs, target_q_vals, q_vals, step):
        for k, idx in enumerate(batch_idxs):

            progress1 = (q_vals[k] - self.contents['state0_val'][idx]) / \
                        (step - self.contents['last_seen'][idx])
            progress2 = (target_q_vals[k] - self.contents['state1_target_val'][idx]) / \
                        (step - self.contents['last_seen'][idx])

            self.update_priority(batch_idxs[k], progress1)
            self.update_priority(batch_idxs[k], progress2)

            self.contents['state1_target_val'][batch_idxs[k]] = target_q_vals[k]
            self.contents['state0_val'][batch_idxs[k]] = q_vals[k]

