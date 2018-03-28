from .wrapper import goal_basic
import numpy as np
from gym.spaces import Box


class HcRootx(goal_basic):
    def __init__(self, env, reward_type):
        super(HcRootx, self).__init__(env, reward_type)
        self.state_to_goal = range(6,12)
        self.state_to_obs = range(12)
        self.state_to_reached = range(6,12)
        self.reward_range = [-0.6, 100]
        self.goal_space = Box(np.array([0]), np.array([100]))
        self.initial_goal = np.array([100])
