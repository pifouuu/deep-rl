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
        self.state_to_goal = [8,9]
        self.state_to_reached = [6,7]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))
        self.initial_goal = np.array([0, 0.1])
        self.start = np.array([1, 1, 0, 0, 0, 0, 0.2, 0.0])


    def _reset(self):
        _ = self.env.reset()
        qpos = self.unwrapped.model.data.qpos.flatten()
        qvel = self.unwrapped.model.data.qvel.flatten()
        qpos[[2,3]] = self.goal
        self.unwrapped.set_state(qpos, qvel)
        obs = self.unwrapped._get_obs()
        self.starts.append(obs)
        state = self.add_goal(obs, self.goal)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()
        return state

    def find_goal_reachable(self):
        while True:
            goal = self.goal_space.sample()
            if np.linalg.norm(goal) < 0.2:
                break
        return goal

    def set_goal_reachable(self):
        self.goal = self.find_goal_reachable()

    def is_reachable(self):
        return (np.linalg.norm(self.goal) < 0.2)

class ReacherEps(Reacher):
    def __init__(self, env, reward_type, epsilon):
        super(ReacherEps, self).__init__(env, reward_type, epsilon)
        self.goal_space = Box(np.array([-0.2, -0.2, 0.01]), np.array([0.2, 0.2, 0.1]))

        #TODO : goal to state

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



