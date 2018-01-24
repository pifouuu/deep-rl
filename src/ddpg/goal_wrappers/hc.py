from .wrapper import goal_basic
import numpy as np
from gym.spaces import Box


class HcRootx(goal_basic):
    def __init__(self, env):
        super(HcRootx, self).__init__(env)
        self.goals = range(100)
        self.state_to_goal = [18]
        self.state_to_obs = range(18)
        self.obs_to_goal = [0]
        self.reward_range = [-0.6, 100]
        self.goal_space = Box(np.array([0]), np.array([100]))
        self.start = np.array([0])
        self.initial_goal = np.array([100])
