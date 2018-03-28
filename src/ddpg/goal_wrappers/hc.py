from .wrapper import goal_basic, no_goal
import numpy as np
from gym.spaces import Box

class HalfCheetahNoGoal(no_goal):
    def __init__(self,env, reward_type):
        super(HalfCheetahNoGoal, self).__init__(env, reward_type)
        self.initial_goal = np.array([100])
        self.state_to_reached = [0]


class HalfCheetahX(goal_basic):
    def __init__(self, env, reward_type):
        super(HalfCheetahX, self).__init__(env, reward_type)
        self.state_to_goal = [18]
        self.state_to_reached = [0]
        self.goal_space = Box(np.array([0]), np.array([100]))
        self.initial_goal = np.array([100])

