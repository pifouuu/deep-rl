import numpy as np
from segmentTree import SumSegmentTree, MinSegmentTree

# added by Olivier Sigaud --------------------------------
# import pickle
# end of added by Olivier Sigaud --------------------------------

#TODO : same buffer for goals and for expe
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

    def dump(self):
        """Get all of the data in a single array"""
        return (self.data[:self.length])


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class ReplayBuffer(object):
    def __init__(self, limit, content_shape):
        self.contents = {}
        self.length = 0
        for content, shape in content_shape.items():
            self.contents[content] = RingBuffer(limit, shape=shape)

    def append(self, buffer_item):
        for name, value in self.contents.items():
            self.length += 1
            value.append(buffer_item[name])

    def dump(self):
        """Get all of the data in a single array"""
        return (self.contents[:self.length])

class Memory(ReplayBuffer):
    def __init__(self, contents_shape, limit):
        super(Memory, self).__init__(limit, contents_shape)

    def end_episode(self):
        pass

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)
        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value.get_batch(batch_idxs))
        return batch_idxs, result

    @property
    def nb_entries(self):
        return len(self.contents['state0'])


class SARSTMemory(Memory):
    def __init__(self, env_wrapper, limit):
        self.contents_shape = {'state0': env_wrapper.state_shape,
                        'action': env_wrapper.action_shape,
                        'state1': env_wrapper.state_shape,
                        'reward': env_wrapper.reward_shape,
                         'terminal': env_wrapper.terminal_shape}

        super(SARSTMemory, self).__init__(self.contents_shape, limit)

    def build_exp(self, state, action, next_state, reward, terminal):
        dict = {'state0': state,
         'action': action,
         'state1': next_state,
         'reward': reward,
         'terminal': terminal}
        return dict


class SASMemory(Memory):
    def __init__(self, env_wrapper, limit):
        self.env_wrapper = env_wrapper
        self.contents_shape = {'state0': env_wrapper.state_shape,
                    'action': env_wrapper.action_shape,
                    'state1': env_wrapper.state_shape}
        super(SASMemory, self).__init__(self.contents_shape, limit)

    def build_exp(self, state, action, next_state, reward, terminal):
        dict = {'state0': state,
                'action': action,
                'state1': next_state}
        return dict

    def sample(self, batch_size):
        _, dict = super(SASMemory, self).sample(batch_size)
        dict['reward'] = np.zeros((batch_size))
        dict['terminal'] = np.zeros((batch_size))
        for k in range(batch_size):
            dict['reward'][k], dict['terminal'][k] = self.env_wrapper.eval_exp(dict['state0'][k],
                                                                               dict['action'][k],
                                                                               dict['state1'][k])
        dict['reward'] = array_min2d(dict['reward'])
        dict['terminal'] = array_min2d(dict['terminal'])
        return dict


    # added by Olivier Sigaud --------------------------------

    # def rewards(self):
    #     vec = self.buffer.contents
    #     return vec['reward'].data
    #
    # # maybe add the other accessors
    #
    # # specific to Continuous Mountain Car, could be generalized
    # def plot2D(self):
    #     plt.figure(1, figsize=(13, 20))
    #     plt.xlabel("position")
    #     plt.ylabel("velocity")
    #     plt.title("Content of the replay buffer")
    #
    #     vec = self.buffer.contents
    #
    #     states = vec['state0'].data
    #     rewards = vec['reward'].data
    #
    #     states = np.array(states)
    #     plt.set_cmap('jet')
    #     plt.scatter(states[:, 0], states[:, 1], s=1, c=rewards)
    #     plt.colorbar(label="rewards")
    #     plt.show()
    #
    # # warning: only saves the content, does not save the parameters such as size_limit, etc.
    # def save(self, file):
    #     """Dump the memory into a pickle file"""
    #     print("Saving memory")
    #     with open(file, "wb") as fd:
    #         pickle.dump(self.dump(), fd)
    #
    # def dump(self):
    #     """Get the memory content as a dictionary"""
    #     return (self.buffer)
    #
    # # warning: only loads the content, does not set the parameters such as size_limit, etc.
    # def load_from_file(self, file):
    #     with open(file, "rb") as fd:
    #         self.buffer = pickle.load(fd)
    #
    # # deals with the shift in position (substracts 0.5 to position)
    # def load_from_ManceronBuffer(self, file):
    #     """
    #     used to load a replay buffer saved under Pierre Manceron's format into a replay buffer of Pierre Fournier's format
    #     """
    #     with open(file, "rb") as fd:
    #         manceron_memory = pickle.load(fd)
    #
    #     for idx in range(len(manceron_memory)):
    #         sample = manceron_memory[idx]
    #         state0 = sample[0]
    #         state1 = sample[3]
    #         buffer_item = {'state0': [state0[0] - 0.5, state0[1]],
    #                        'action': sample[1],
    #                        'reward': sample[2],
    #                        'state1': [state1[0] - 0.5, state1[1]],
    #                        'terminal1': sample[4]}
    #         self.append(buffer_item)
            # end of added by Olivier Sigaud --------------------------------


class EpisodicHerSARSTMemory(SARSTMemory):
    def __init__(self, env_wrapper, limit, strategy):
        """Replay buffer that does Hindsight Experience Replay
        obs_to_goal is a function that converts observations to goals
        goal_slice is a slice of indices of goal in observation
        """
        super(EpisodicHerSARSTMemory, self).__init__(env_wrapper, limit)

        self.strategy = strategy
        self.data = []
        self.env_wrapper = env_wrapper
        self.state_to_goal = env_wrapper.state_to_goal
        self.state_to_obs = env_wrapper.state_to_obs
        self.obs_to_goal = env_wrapper.obs_to_goal

    def append(self, buffer_item):
        super(EpisodicHerSARSTMemory, self).append(buffer_item)
        self.data.append(buffer_item)

    def change_goal(self, buffer_item, new_goal):
        res = buffer_item
        res['state0'][self.state_to_goal] = new_goal
        res['state1'][self.state_to_goal] = new_goal
        res['reward'], res['terminal'] = self.env_wrapper.eval_exp(res['state0'],
                                                                   res['action'],
                                                                   res['state1'])
        return res

    def end_episode(self):
        if self.strategy == 'final':
            final_state = self.data[-1]['state1']
            new_goal = final_state[self.state_to_obs][self.obs_to_goal]
            for buffer_item in self.data:
                new_buffer_item = self.change_goal(buffer_item, new_goal)
                super(EpisodicHerSARSTMemory, self).append(new_buffer_item)
        elif self.strategy == 'episode':
            final_states = [self.data[i]['state1'] for i in list(np.random.randint(len(self.data), size=4))]
            for final_state in final_states:
                new_goal = final_state[self.state_to_obs][self.obs_to_goal]
                for buffer_item in self.data:
                    new_buffer_item = self.change_goal(buffer_item, new_goal)
                    super(EpisodicHerSARSTMemory, self).append(new_buffer_item)
        elif self.strategy == 'future':
            for idx, buffer_item in enumerate(self.data):
                future_indices = np.random.randint(idx, len(self.data), size=4)
                final_states = [self.data[i]['state1'] for i in list(future_indices)]
                for final_state in final_states:
                    new_goal = final_state[self.state_to_obs][self.obs_to_goal]
                    new_buffer_item = self.change_goal(buffer_item, new_goal)
                    super(EpisodicHerSARSTMemory, self).append(new_buffer_item)
        else:
            print('error her strategy')
            return
        self.data = []

class EpisodicHerSASMemory(SASMemory):
    def __init__(self, env_wrapper, limit, strategy):
        super(EpisodicHerSASMemory, self).__init__(env_wrapper, limit)

        self.strategy = strategy
        self.data = []
        self.state_to_goal = env_wrapper.state_to_goal
        self.state_to_obs = env_wrapper.state_to_obs
        self.obs_to_goal = env_wrapper.obs_to_goal

    def append(self, buffer_item):
        super(EpisodicHerSASMemory, self).append(buffer_item)
        self.data.append(buffer_item)

    def change_goal(self, buffer_item, new_goal):
        res = buffer_item
        res['state0'][self.state_to_goal] = new_goal
        res['state1'][self.state_to_goal] = new_goal
        return res

    def end_episode(self):
        if self.strategy == 'final':
            final_state = self.data[-1]['state1']
            new_goal = final_state[self.state_to_obs][self.obs_to_goal]
            for buffer_item in self.data:
                new_buffer_item = self.change_goal(buffer_item, new_goal)
                super(EpisodicHerSASMemory, self).append(new_buffer_item)
        elif self.strategy == 'episode':
            final_states = [self.data[i]['state1'] for i in list(np.random.randint(len(self.data), size=4))]
            for final_state in final_states:
                new_goal = final_state[self.state_to_obs][self.obs_to_goal]
                for buffer_item in self.data:
                    new_buffer_item = self.change_goal(buffer_item, new_goal)
                    super(EpisodicHerSASMemory, self).append(new_buffer_item)
        elif self.strategy == 'future':
            for idx, buffer_item in enumerate(self.data):
                future_indices = np.random.randint(idx, len(self.data), size=4)
                final_states = [self.data[i]['state1'] for i in list(future_indices)]
                for final_state in final_states:
                    new_goal = final_state[self.state_to_obs][self.obs_to_goal]
                    new_buffer_item = self.change_goal(buffer_item, new_goal)
                    super(EpisodicHerSASMemory, self).append(new_buffer_item)
        else:
            print('error her strategy')
            return
        self.data = []







