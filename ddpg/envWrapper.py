import numpy as np
import math

class EnvWrapper(object):
    def __init__(self):
        self.eps = 0.1
        self.obs_to_goal = [0]
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.min_reward = -0.1
        self.max_reward = 100

    def get_initial_goal(self):
        goal = np.random.uniform([0.449], [0.451], (1,))
        return goal

    def get_random_goal(self):
        goal_found = False
        goal = None
        while not goal_found:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            goal_found = np.abs(goal+0.5) > self.eps
        return goal

class NoGoalWrapper(EnvWrapper):
    def __init__(self):
        super(NoGoalWrapper, self).__init__()
        self.state_shape = (2,)
        self.has_goal = False

    def eval_exp(self, state0, action, state1):
        r = 0
        term = False
        if state1[self.obs_to_goal] >= 0.45:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

class WithGoal(EnvWrapper):
    def __init__(self):
        super(WithGoal, self).__init__()
        self.state_to_obs = [0, 1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.has_goal = True

    def eval_exp(self, state0, action, state1):
        r = 0
        term = False
        if np.abs(state1[self.state_to_obs][self.obs_to_goal] -
                          state1[self.state_to_goal]) < self.eps:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term


class IntervalCurriculum(WithGoal):
    def __init__(self):
        super(IntervalCurriculum, self).__init__()
        self.disc_goal_space = list(np.arange(-1.2, 0.7, 0.1))
        self.difficulties = np.array([6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # self.means = np.linspace(4, 10, 100)
        # self.stds = np.linspace(0.5, 2, 100)

    def get_intervals(self):
        starts = self.disc_goal_space[:-1]
        ends = self.disc_goal_space[1:]
        intervals = [np.array([a,b]) for a,b in zip(starts,ends)]
        return intervals

    def get_priorities(self):
        return self.difficulties

    # def sample_goal(self, obs, successes):
    #     mean = self.means[min(successes, 99)]
    #     std = self.stds[min(successes, 99)]
    #     difficulty = np.clip(int(np.random.normal(mean,std)), 1, 10)
    #     indices = np.squeeze(np.argwhere(self.difficulties==difficulty), axis=1)
    #     idx = np.random.choice(indices)
    #     goal = np.random.uniform(self.disc_goal_space[idx], self.disc_goal_space[idx+1], (1,))
    #     return goal


class GoalCurriculum(WithGoal):
    def __init__(self):
        super(GoalCurriculum, self).__init__()
        self.goals = list(np.linspace(-1.2, 0.6, 100))

    def get_goals(self):
        return self.goals

    def get_priorities(self):
        distances = [np.abs(g + 0.5) for g in self.goals]
        difficulties = [d if d >= 0.1 else 0 for d in distances]
        return difficulties
