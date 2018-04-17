from .wrapper import goal_basic, no_goal
import numpy as np

from gym.spaces import Box

class ReacherNoGoal(no_goal):
    def __init__(self, env, reward_type, epsilon):
        super(ReacherNoGoal, self).__init__(env, reward_type, epsilon)
        self.initial_goal = np.array([0, 0.1])
        self.state_to_reached = [6, 7]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))

class Reacher(goal_basic):
    def __init__(self, env, reward_type, epsilon):
        super(Reacher, self).__init__(env, reward_type, epsilon)
        self.internal = [8,9]
        self.state_to_goal = [8,9]
        self.state_to_reached = [6,7]
        self.goal_to_target = [0,1]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))
        self.initial_goal = np.array([0, 0.1])
        self.start = np.array([1, 1, 0, 0, 0, 0, 0.2, 0.0])
        self.dims_curri = [8,9]


    def _reset(self):
        _ = self.env.reset()
        qpos = self.unwrapped.model.data.qpos.flatten()
        qvel = self.unwrapped.model.data.qvel.flatten()
        qpos[[2,3]] = self.goal[self.goal_to_target]
        self.unwrapped.set_state(qpos, qvel)
        obs = self.unwrapped._get_obs()
        state = self.add_goal(obs, self.goal)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()
        return state

    def sample_goal(self):
        while True:
            goal = np.random.uniform(self.goal_space.low, self.goal_space.high)
            if np.linalg.norm(goal[self.goal_to_target]) < 0.2:
                break
        return goal

    def sample_test_goal(self):
        return self.sample_goal()

    def is_reachable(self):
        return (np.linalg.norm(self.goal[self.goal_to_target]) < 0.2)

class ReacherEps_x_y(Reacher):
    def __init__(self, env, reward_type, epsilon):
        super(ReacherEps_x_y, self).__init__(env, reward_type, epsilon)
        self.goal_space = Box(np.array([-0.2, -0.2, 0.01]), np.array([0.2, 0.2, 0.1]))
        self.initial_goal = np.array([0, 0.1, epsilon])
        self.internal = [8, 9, 10]
        self.internal_to_epsilon = [10]
        self.dims_curri = [8,9]

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        goal_reached = agent_state_1[self.state_to_reached]
        goal = agent_state_1[self.state_to_goal]
        vec = goal - goal_reached
        d = np.linalg.norm(vec)
        term = d < agent_state_1[self.internal_to_epsilon]
        r = 0
        if not term:
            if self.reward_type == 'sparse':
                r = -1
            elif self.reward_type == 'dense':
                r = - d
        return r, term

    def sample_test_goal(self):
        goal = self.sample_goal()
        goal[[2]] = self.initial_goal[[2]]
        return goal

class ReacherEps_e(ReacherEps_x_y):
    def __init__(self, env, reward_type, epsilon):
        super(ReacherEps_e, self).__init__(env, reward_type, epsilon)
        self.dims_curri = [10]

class ReacherEps_x_y_e(ReacherEps_x_y):
    def __init__(self, env, reward_type, epsilon):
        super(ReacherEps_x_y_e, self).__init__(env, reward_type, epsilon)
        self.dims_curri = [8,9,10]

class ReacherOrigin(Reacher):
    def __init__(self, env, reward_type, epsilon):
        super(ReacherOrigin, self).__init__(env, reward_type, epsilon)

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        vec = agent_state_1[self.state_to_reached]
        # widths = self.goal_space.high - self.goal_space.low
        # vec = np.divide(vec, widths)
        dist = np.linalg.norm(vec)
        ctrl = np.square(action).sum()
        r = - dist - ctrl
        return r, False



