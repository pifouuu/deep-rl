import numpy as np
import random as rnd
# from matplotlib import pyplot as plt

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
        idx = (self.start + self.length - 1) % self.maxlen
        self.data[idx] = v
        # return idx

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
            value.append(buffer_item[name])
        # return idx

    def dump(self):
        """Get all of the data in a single array"""
        return (self.contents[:self.length])

class Memory(ReplayBuffer):
    def __init__(self, contents_shape, limit):
        super(Memory, self).__init__(limit, contents_shape)

    def end_episode(self, goal_reached):
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
    def __init__(self, env, limit):
        self.contents_shape = {'state0': env.state_dim,
                        'action': env.action_dim,
                        'state1': env.state_dim,
                        'reward': (1,),
                         'terminal': (1,)}

        super(SARSTMemory, self).__init__(self.contents_shape, limit)
        self.env = env

    def build_exp(self, state, action, next_state, reward, terminal):
        dict = {'state0': state,
         'action': action,
         'state1': next_state,
         'reward': reward,
         'terminal': terminal}
        return dict

class EpisodicHerSARSTMemory(SARSTMemory):
    def __init__(self, env, limit, strategy, n_her_goals):
        """Replay buffer that does Hindsight Experience Replay
        obs_to_goal is a function that converts observations to goals
        goal_slice is a slice of indices of goal in observation
        """
        super(EpisodicHerSARSTMemory, self).__init__(env, limit)

        self.strategy = strategy
        self.n_her_goals = n_her_goals
        self.data = []

    def append(self, buffer_item):
        super(EpisodicHerSARSTMemory, self).append(buffer_item)
        self.data.append(buffer_item)

    def end_episode(self, goal_reached):

        cur_t = self.data[-1]['state0'][[8, 9]]
        cur_eps = self.data[0]['state0'][[10]]
        epsilons = [cur_eps]
        targets = [[cur_t] for _ in self.data]

        cur_idx = self.env.eps.index(cur_eps)
        strat_e = self.strategy.split('_')[1]
        if strat_e == 'easier':
            epsilons += [eps for eps in self.env.eps if eps > cur_eps]
        elif strat_e == 'harder':
            epsilons += [eps for eps in self.env.eps if eps < cur_eps]
        elif strat_e == 'all':
            epsilons += [eps for eps in self.env.eps if eps != cur_eps]
        elif strat_e == '1':
            if cur_idx < 3:
                epsilons.append(self.env.eps[cur_idx + 1])
            if cur_idx > 0:
                epsilons.append(self.env.eps[cur_idx - 1])
        elif strat_e == 'easier1':
            if cur_idx < 3:
                epsilons.append(self.env.eps[cur_idx + 1])
        elif strat_e == 'harder1':
            if cur_idx > 0:
                epsilons.append(self.env.eps[cur_idx - 1])

        strat_xy = self.strategy.split('_')[0]
        if strat_xy == 'future':
            for idx, buffer_item in enumerate(self.data):
                indices = range(idx, len(self.data))
                future_indices = rnd.sample(indices, np.min([self.n_her_goals, len(indices)]))
                targets[idx] += [self.data[i]['state1'][[6,7]] for i in list(future_indices)]
        if strat_xy == 'no':
            pass
        if strat_xy == 'final':
            target = self.data[-1]['state1'][[6,7]]
            for idx, _ in enumerate(self.data):
                targets[idx] += [target]

        for idx, buffer_item in enumerate(self.data):
            t = targets[idx]
            eps = epsilons
            for new_t in t:
                for new_eps in eps:
                    if (new_t != cur_t).any() or new_eps != cur_eps:
                        res = buffer_item.copy()
                        res['state0'][[8, 9]] = new_t
                        res['state1'][[8, 9]] = new_t
                        res['state0'][[10]] = new_eps
                        res['state1'][[10]] = new_eps
                        res['reward'], res['terminal'] = self.env.eval_exp(res['state0'],
                                                                           res['action'],
                                                                           res['state1'],
                                                                           res['reward'],
                                                                           res['terminal'])
                        super(EpisodicHerSARSTMemory, self).append(res)

        self.data = []


