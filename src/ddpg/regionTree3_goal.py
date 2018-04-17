import numpy as np
from ddpg.regions import Queue

class FixedGoalMemory():
    def __init__(self, space, dims, buffer, actor, critic, N, n_split, split_min, alpha, maxlen, n_window, render, sampler):
        self.maxlen = maxlen
        self.n_window = n_window
        self.alpha = alpha
        self.dims = dims
        self.space = space

        self.buffer = buffer
        self.sampler = sampler

        self.actor = actor
        self.critic = critic

        self.goal_set = []
        self.goal_queues = []
        self.goal_freq = []

        self.n_goals = N
        self.initialize()


    def initialize(self):
        obs_dummy = self.buffer.env.observation_space.low
        self.goal_set = [np.concatenate([obs_dummy, self.buffer.env.sample_goal_reachable('uni','uni')])
                         for _ in range(self.n_goals)]
        self.goal_queues = [Queue(maxlen=self.maxlen, n_window=self.n_window) for _ in range(self.n_goals)]
        self.goal_freq = [0 for _ in range(self.n_goals)]

    def end_episode(self, goal_reached):
        self.buffer.end_episode(goal_reached)
        if self.buffer.env.goal_parameterized:
            self.update_competence()
            for queue in self.goal_queues:
                queue.update_CP()


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
        for idx in range(self.n_goals):
            goal = []
            for dim in self.buffer.env.internal:
                if dim in self.dims:
                    val_dim = self.goal_set[idx][dim]
                else:
                    rnd = np.random.randint(self.n_goals)
                    val_dim = np.linspace(self.space.low[dim], self.space.high[dim], self.n_goals)[rnd]
                goal.append(val_dim)
            goal = np.array(goal)
            competence = self.eval_goal(goal)
            self.goal_queues[idx].points.append((goal,competence))

    def sample_prop_idx(self):
        CPs = [queue.CP for queue in self.goal_queues]
        sum = np.sum(CPs)
        mass = np.random.random() * sum
        idx = 0
        s = CPs[0]
        while mass > s:
            idx += 1
            s += CPs[idx]
        return idx



    def sample_goal(self):
        obs_dummy = self.buffer.env.observation_space.low
        m = obs_dummy.shape[0]
        if self.n_goals != 0:
            if np.random.random() > self.alpha:
                idx = self.sample_prop_idx()
            else:
                idx = np.random.randint(self.n_goals)
            self.goal_freq[idx] += 1
            drawn_goal = self.goal_set[idx]
        else:
            drawn_goal = np.concatenate([obs_dummy, self.buffer.env.sample_goal_reachable('uni','uni')])

        goal = []

        for dim in self.buffer.env.internal:
            if dim in self.dims:
                val_dim = drawn_goal[dim]
            else:
                if self.sampler == 'init':
                    val_dim = self.buffer.env.initial_goal[dim-m]
                elif self.sampler == 'disc':
                    rnd = np.random.randint(self.n_goals)
                    val_dim = np.linspace(self.space.low[dim], self.space.high[dim], self.n_goals)[rnd]
                elif self.sampler == 'uni':
                    val_dim = np.random.uniform(self.space.low[dim], self.space.high[dim])
                else:
                    raise RuntimeError
            goal.append(val_dim)
        goal = np.array(goal)
        return goal

    def stats(self):
        stats = {}
        stats['list_CP'] = self.list_CP
        stats['list_comp'] = self.list_competence
        stats['list_freq'] = self.goal_freq
        stats['max_CP'] = self.max_CP
        stats['min_CP'] = self.min_CP
        stats['max_comp'] = self.max_competence
        stats['min_comp'] = self.min_competence
        return stats

    @property
    def list_CP(self):
        return [queue.CP for queue in self.goal_queues]

    @property
    def list_competence(self):
        return [queue.competence for queue in self.goal_queues]

    @property
    def min_CP(self):
        try:
            res = np.min(self.list_CP)
        except:
            res = 0
        return res

    @property
    def max_CP(self):
        try:
            res = np.max(self.list_CP)
        except:
            res = 0
        return res

    @property
    def max_competence(self):
        try:
            res = np.max(self.list_competence)
        except:
            res = 0
        return res

    @property
    def min_competence(self):
        try:
            res = np.min(self.list_competence)
        except:
            res = 0
        return res
