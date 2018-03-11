from .wrapper import goal_basic
import numpy as np
import six


from gym.spaces import Box

class Manipulator(goal_basic):
    def __init__(self, env):
        super(Manipulator, self).__init__(env)
        self.goals = []
        # TODO a changer
        self.state_to_goal = range(33,37)
        self.state_to_obs = range(33)
        self.state_to_reached = [6,7]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([-0.2, 0.2]))
        self.initial_goal = np.array([0, 0.1])
        self.reward_range = [-0.2, 100]

        self.target = 'target_ball'

    def _reset(self):
        _ = self.env.reset()

        # Randomise target location
        # TODO: integrate possibility to have target in receptacle more easily
        target_x = self.goal[0]
        target_z = self.goal[1]
        target_angle = self.goal[2]

        target_idx = self.unwrapped.model.body_names.index(six.b(self.target))
        body_pos = self.unwrapped.model.body_pos.copy()
        body_pos[target_idx, [0, 2]] = target_x, target_z
        body_quat = self.unwrapped.model.body_quat.copy()
        body_quat[target_idx, [0, 2]] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]

        self.unwrapped.model.body_quat = body_quat
        self.unwrapped.model.body_pos = body_pos

        return self.unwrapped._get_obs()

    def get_random_goal(self):
        target_x = np.random.uniform(-.4, .4)
        target_z = np.random.uniform(.1, .4)
        target_angle = np.random.uniform(-np.pi, np.pi)
        return [target_x, target_z, target_angle]