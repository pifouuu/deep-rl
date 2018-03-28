from .wrapper import goal_basic, no_goal
import numpy as np
from gym.spaces import Box

class CmcNoGoal(no_goal):
    def __init__(self, env, reward_type):
        super(CmcNoGoal, self).__init__(env, reward_type)
        self.initial_goal = np.array([0.45])
        self.state_to_reached = [0]

class CmcPos(goal_basic):
    def __init__(self, env, reward_type):
        super(CmcPos, self).__init__(env, reward_type)
        self.state_to_goal = [2]
        self.state_to_reached = [0]
        self.goal_space = Box(np.array([-1.2]), np.array([0.6]))
        self.initial_goal = np.array([0.45])

    def _reset(self):
        state = super(CmcPos, self)._reset()
        self.unwrapped.goal_position = self.goal
        return state

    # def set_goal_rnd(self):
    #     while True:
    #         goal = self.goal_space.sample()
    #         if np.abs(goal+0.5) > 0.1: break
    #     self.goal = goal

class CmcFull(goal_basic):
    def __init__(self, env, reward_type):
        super(CmcFull, self).__init__(env, reward_type)
        self.state_to_goal = [2,3]
        self.state_to_reached = [0,1]
        self.goal_space = Box(np.array([-1.2, -0.07]), np.array([0.6, 0.07]))
        self.initial_goal = np.array([0.45, 0])



