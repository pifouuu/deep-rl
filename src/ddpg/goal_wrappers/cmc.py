from .wrapper import goal_basic
import numpy as np

class CmcPos(goal_basic):
    def __init__(self, env):
        super(CmcPos, self).__init__(env)
        self.goals = list(np.linspace(-1.2, 0.6, 100))
        self.state_to_goal = [2]
        self.state_to_obs = range(2)
        self.obs_to_goal = [0]
        self.low = [-1.2]
        self.high = [0.6]
        self.start = np.array([-0.5])
        self.initial_goal = np.array([0.45])
        self.reward_range = [-0.1, 100]


class CmcFull(goal_basic):
    def __init__(self, env):
        super(CmcFull, self).__init__(env)
        grids = np.meshgrid(np.linspace(-1.2,0.6,10),np.linspace(-0.07,0.07,10))
        self.goals = list(np.array(grids).reshape(2, -1).T)
        self.state_to_goal = [2,3]
        self.state_to_obs = range(2)
        self.obs_to_goal = [0,1]
        self.low = [-1.2, -0.07]
        self.high = [0.6, 0.07]
        self.start = np.array([-0.5,0])
        self.initial_goal = np.array([0.45, 0])
        self.reward_range = [-0.1, 100]



