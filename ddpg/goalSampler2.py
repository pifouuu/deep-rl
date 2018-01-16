from ddpg.segmentTree import SumSegmentTree, MinSegmentTree
import numpy as np

class GoalSampler():
    def __init__(self, env):
        self.env = env
        self.stats = {}

class RandomGoalSampler(GoalSampler):
    def __init__(self, env):
        super(RandomGoalSampler, self).__init__(env)

    def sample(self):
        return self.env.get_random_goal()

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


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Buffer(GoalSampler):
    def __init__(self, limit, content_shape, env_wrapper):
        super(Buffer, self).__init__(env_wrapper)
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

class PrioritizedBuffer(Buffer):
    def __init__(self, limit, alpha, content, env_wrapper):
        super(PrioritizedBuffer, self).__init__(limit, content, env_wrapper)
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

    def sample(self):
        # Draw such that we always have a proceeding element.
        idx = self.sample_proportional_idx()
        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value[idx])
        return idx, result

    def update_priority(self, idx, priority):
        self._it_sum[idx] = np.clip(priority ** self.alpha, self._min_priority, self._max_priority)

class PrioritizedGoalBuffer(PrioritizedBuffer):
    def __init__(self, limit, alpha, env):
        self.contents_shape = {'goal':(self.env.goal_dim,)}
        super(PrioritizedGoalBuffer,self).__init__(limit, alpha, self.contents_shape, env)
        self.goals = env.goals()
        self.priorities = env.difficulties()
        for goal, priority in zip(self.goals, self.priorities):
            buffer_item = {'goal': goal}
            self.append(buffer_item, priority)

    def sample(self):
        sample_idx, sample_dict = super().sample()
        goal = sample_dict['goal']
        return np.reshape(goal, (self.env.goal_dim,))

class CompetenceProgressGoalBuffer(PrioritizedBuffer):
    def __init__(self, limit, alpha, env, actor, critic):
        self.contents_shape = {'goal':(self.env.goal_dim,)}
        super(CompetenceProgressGoalBuffer, self).__init__(limit, alpha, self.contents_shape, env)
        self.goals = env.goals()
        self.competences = [0]*len(self.goals)
        self.progresses = [0]*len(self.goals)
        self.actor = actor
        self.critic = critic
        self.nb_sampled = 0
        for goal in self.goals:
            buffer_item = {'goal': goal}
            self.append(buffer_item, 1)

    def update_competence(self):
        starts = np.random.uniform(low=-0.6, high=-0.4, size=10)
        states = np.array([[start, 0, goal] for goal in self.goals for start in starts])
        a_outs = self.actor.predict(states)
        q_outs = list(self.critic.predict(states, a_outs))
        q_mean_outs = [np.array(q_outs[k:k+10]).mean() for k in range(0,1000,10)]
        return q_mean_outs

    def sample(self):
        if self.nb_sampled % 10 == 0:
            new_competences = self.update_competence()
            self.progresses = [a - b for (a, b) in zip(new_competences, self.competences)]
            for idx, progress in enumerate(self.progresses):
                self.update_priority(idx, np.abs(progress))
            self.competences = new_competences
            self.stats['q_values'] = self.competences
            self.stats['d_q_values'] = self.progresses

        sample_idx, sample_dict = super(CompetenceProgressGoalBuffer, self).sample()
        goal = sample_dict['goal']
        self.nb_sampled += 1
        return np.reshape(goal,(self.env.goal_dim,))
