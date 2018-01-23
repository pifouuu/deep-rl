from gym import Wrapper
import numpy as np
import math

class no_goal(Wrapper):
    def __init__(self, env):
        super(no_goal, self).__init__(env)
        self.goal = []
        self.state_to_goal = []
        self.state_to_obs = range(env.observation_space.high.shape[0])

    def eval_exp(self, previous_state_goal, action, state_goal, reward, terminal):
        return reward, terminal

    def get_random_goal(self):
        return []

    def get_initial_goal(self):
        return []

    def add_goal(self, state, goal):
        return state

class goal_basic(Wrapper):
    def __init__(self,env):
        super(goal_basic, self).__init__(env)
        self.goal = []
        self.state_to_goal = []
        self.obs_to_goal = []
        self.state_to_obs = []
        self.low = -np.inf
        self.high = np.inf
        self.start = np.array([])
        self.initial_goal = np.array([])
        self.reward_range = [0, 0]

    def add_goal(self, state, goal):
        return np.concatenate([state, goal])

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
            goal = np.random.uniform(self.low, self.high, (1,))
            if np.linalg.norm(goal - self.start) > 0.05:
                break
        return goal

    def get_initial_goal(self):
        return self.initial_goal

    def change_goal(self, buffer_item, final_state):
        res = buffer_item
        res['state0'][self.state_to_goal] = final_state[self.state_to_obs][self.obs_to_goal]
        res['state1'][self.state_to_goal] = final_state[self.state_to_obs][self.obs_to_goal]
        res['reward'], res['terminal'] = self.eval_exp(res['state0'],
                                                           res['action'],
                                                           res['state1'],
                                                           res['reward'],
                                                           res['terminal'])
        return res
