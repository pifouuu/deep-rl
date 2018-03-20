from .wrapper import goal_basic
import numpy as np
from gym.spaces import Box

class CmcPos(goal_basic):
    def __init__(self, env):
        super(CmcPos, self).__init__(env)
        self.state_to_goal = [2]
        self.state_to_obs = range(2)
        self.state_to_reached = [0]
        self.goal_space = Box(np.array([-1.2]), np.array([0.6]))
        self.initial_goal = np.array([0.45])
        self.reward_range = [-0.1, 100]

    def set_goal_init(self):
        self.goal = np.random.uniform([0.449], [0.451], (1,))

    def get_start(self):
        return np.array([np.random.uniform(low=-0.6, high=-0.4), 0])

    def _reset(self):
        obs = self.env.reset()
        self.unwrapped.goal_position = self.goal
        state = self.add_goal(obs, self.goal)
        self.prev_state = state
        return state

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = 0
        goal_reached = agent_state_1[self.state_to_reached]
        goal = agent_state_1[self.state_to_goal]
        vec = goal - goal_reached
        term = np.linalg.norm(vec) < 0.1
        if term:
            r += 100
        r -= 0.1 * np.square(action).sum()
        return r, term

class CmcFull(goal_basic):
    def __init__(self, env):
        super(CmcFull, self).__init__(env)
        grids = np.meshgrid(np.linspace(-1.2,0.6,10),np.linspace(-0.07,0.07,10))
        self.goals = list(np.array(grids).reshape(2, -1).T)
        self.state_to_goal = [2,3]
        self.state_to_obs = range(2)
        self.state_to_reached = [0,1]
        self.goal_space = Box(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
        self.initial_goal = np.array([0.45, 0])
        self.reward_range = [-0.1, 100]



