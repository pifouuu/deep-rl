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

class Buffer(object):
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

        self.buffer = Buffer(limit, contents)
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

