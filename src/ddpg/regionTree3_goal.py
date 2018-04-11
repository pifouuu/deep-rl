import numpy as np
from ddpg.regions import Queue

class FixedGoalMemory():
    def __init__(self, space, dims, buffer, actor, critic, N, n_split, split_min, alpha, maxlen, n_window, render, sampler):
        self.maxlen = maxlen
        self.n_window = n_window
        self.alpha = alpha

        self.buffer = buffer
        self.sampler = sampler

        self.actor = actor
        self.critic = critic

        self.goal_set = []
        self.goal_queues = []

        self.n_goals = N
        self.initialize()


    def initialize(self):
        self.goal_set = [self.buffer.env.find_goal_reachable() for _ in range(self.n_goals)]
        self.goal_queues = [Queue(maxlen=self.maxlen, n_window=self.n_window) for _ in range(self.n_goals)]

    def end_episode(self, goal_reached):
        self.buffer.end_episode(goal_reached)
        if self.buffer.env.goal_parameterized:
            self.update_competence()

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
        for goal,queue in zip(self.goal_set, self.goal_queues):
            competence = self.eval_goal(goal)
            queue.points.append((goal,competence))

    def sample_prop_goal(self):
        CPs = [queue.CP for queue in self.goal_queues]
        sum = np.sum(CPs)
        mass = np.random.random() * sum
        idx = 0
        s = CPs[0]
        while mass > s:
            idx += 1
            s += CPs[idx]
        return self.goal_set[idx]

    def sample_random_goal(self):
        idx = np.random.randint(self.n_goals)
        goal = self.goal_set[idx]
        return goal

    def sample_goal(self):
        if self.sampler=='init':
            return self.buffer.env.initial_goal
        elif self.sampler=='rnd':
            return self.sample_random_goal()
        elif self.sampler=='prio':
            p = np.random.random()
            if p < self.alpha:
                return self.sample_random_goal()
            else:
                return self.sample_prop_goal()
        else:
            raise RuntimeError

    def stats(self):
        stats = {}
        stats['max_CP'] = self.max_CP
        stats['min_CP'] = self.min_CP
        stats['max_comp'] = self.max_competence
        stats['min_comp'] = self.min_competence
        return stats

    @property
    def min_CP(self):
        return np.min([queue.CP for queue in self.goal_queues])

    @property
    def max_CP(self):
        return np.max([queue.CP for queue in self.goal_queues])

    @property
    def max_competence(self):
        return np.max([queue.competence for queue in self.goal_queues])

    @property
    def min_competence(self):
        return np.min([queue.competence for queue in self.goal_queues])