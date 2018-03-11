from .wrapper import goal_basic
import numpy as np

from gym.spaces import Box

class ReacherSparse(goal_basic):
    def __init__(self, env):
        super(ReacherSparse, self).__init__(env)
        self.goals = []
        self.state_to_goal = [8,9]
        self.state_to_obs = [0, 1, 2, 3, 4, 5, 6, 7]
        self.state_to_reached = [6,7]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))
        self.initial_goal = np.array([0, 0.1])
        self.reward_range = [-0.2, 100]
        for _ in range(100):
            while True:
                pt = np.random.uniform(low=-.2, high=.2, size=2)
                if np.linalg.norm(pt) < 0.2: break
            self.goals.append(pt)

    def _reset(self):
        _ = self.env.reset()
        qpos = self.unwrapped.model.data.qpos.flatten()
        qvel = self.unwrapped.model.data.qvel.flatten()
        qpos[[2,3]] = self.goal
        self.unwrapped.set_state(qpos, qvel)
        return self.unwrapped._get_obs()






