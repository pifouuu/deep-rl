from gym import Wrapper
import numpy as np
import math

class goal_wrapper(Wrapper):
    def __init__(self, env):
        super(goal_wrapper, self).__init__(env)
        self.state_to_goal = []
        self.state_to_obs = range(env.observation_space.high.shape[0])

    # def _reset(self):
    #     self.goal = self.goal_sampler.sample()
    #     state = self.env.reset()
    #     state_goal = np.concatenate([state,self.goal])
    #     self.previous_state_goal = state_goal
    #     return state_goal
    #
    # def _step(self, action):
    #     state, reward, terminal, info = self.env.step(action)
    #     state_goal = np.concatenate([state, self.goal])
    #     reward, terminal = self.eval_exp(self.previous_state_goal, action, state_goal)
    #     self.previous_state_goal = state_goal
    #     return state_goal, reward, terminal, info

    def eval_exp(self, previous_state_goal, action, state_goal, reward, terminal):
        return reward, terminal

    def get_random_goal(self):
        return []

    def get_initial_goal(self):
        return []

class HalfCheetahEnvGoal(goal_wrapper):
    def __init__(self, env):
        super(HalfCheetahEnvGoal, self).__init__(env)
        self.goals = range(100)
        self.state_to_goal = [18]
        self.state_to_obs = range(18)
        self.obs_to_goal = [0]
        self.difficulties = range(100)
        self.goal_dim = len(self.obs_to_goal)
        self.reward_range = [-0.6, 100]


    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = 0
        goal_reached = agent_state_1[self.obs_to_goal]
        goal = agent_state_1[self.state_to_goal]
        term = goal_reached > goal
        if term:
            r += 100
        r -= 0.1 * np.square(action).sum()
        return r, term

    def get_random_goal(self):
        return np.random.uniform([0], [100], (1,))

    def get_initial_goal(self):
        return [100]

class ReacherEnvGoal(goal_wrapper):
    def __init__(self, env):
        super(ReacherEnvGoal, self).__init__(env)
        self.goals = []
        self.difficulties = []
        self.state_to_goal = [9, 10, 11]
        self.state_to_obs = range(9)
        self.obs_to_goal = [6, 7, 8]
        self.goal_dim = len(self.obs_to_goal)
        self.min_reward = -0.2
        self.max_reward = 100
        for _ in range(100):
            while True:
                pt = np.random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(pt) < 2: break
            self.goals.append(pt)
            pos_init = np.array([0.2, 0])
            dist_init = np.linalg.norm(pos_init - pt)
            self.difficulties.append(dist_init)

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = 0
        goal_reached = agent_state_1[self.obs_to_goal]
        goal = agent_state_1[self.state_to_goal]
        vec = goal - goal_reached
        term = np.linalg.norm(vec) < 0.05
        if term:
            r += 100
        r -= 0.1 * np.square(action).sum()
        return r, term

    def get_random_goal(self):
        while True:
            goal = np.random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(goal) < 2: break
        goal = np.concatenate([goal,[0]])
        return goal

    def get_initial_goal(self):
        return [0, 0.1, 0]


class Continuous_MountainCarEnvGoal(goal_wrapper):
    def __init__(self, env):
        super(Continuous_MountainCarEnvGoal, self).__init__(env)
        self.goals = list(np.linspace(-1.2, 0.6, 100))
        self.state_to_goal = [2]
        self.state_to_obs = range(2)
        self.obs_to_goal = [0]
        distances = [np.abs(g + 0.5) for g in self.goals]
        self.difficulties = [d if d >= 0.1 else 0 for d in distances]
        self.goal_dim = len(self.obs_to_goal)

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = 0
        goal_reached = agent_state_1[self.obs_to_goal]
        goal = agent_state_1[self.state_to_goal]
        term = (goal>-0.5 and goal_reached>goal) or (goal<-0.5 and goal_reached<goal)
        if term:
            r += 100
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def get_random_goal(self):
        while True:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            if np.abs(goal+0.5) > 0.1: break
        return goal

    def get_initial_goal(self):
        return [0.45]
