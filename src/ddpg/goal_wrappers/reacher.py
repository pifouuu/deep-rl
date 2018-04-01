from .wrapper import goal_basic, no_goal
import numpy as np

from gym.spaces import Box

class ReacherNoGoal(no_goal):
    def __init__(self, env, reward_type):
        super(ReacherNoGoal, self).__init__(env, reward_type)
        self.initial_goal = np.array([0, 0.1])
        self.state_to_reached = [6, 7]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))

class Reacher(goal_basic):
    def __init__(self, env, reward_type):
        super(Reacher, self).__init__(env, reward_type)
        self.state_to_goal = [8,9]
        self.state_to_reached = [6,7]
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))
        self.initial_goal = np.array([0, 0.1])

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

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        goal_reached = agent_state_1[self.state_to_reached]
        goal = agent_state_1[self.state_to_goal]
        vec = goal - goal_reached
        widths = self.goal_space.high - self.goal_space.low
        norm_vec = np.divide(vec, widths)
        dist = np.linalg.norm(norm_vec)
        term = dist < self.epsilon
        r=0
        if not term:
            if self.reward_type == 'sparse':
                r = -1
            elif self.reward_type == 'dense':
                r = - dist
        return r, False

    def set_goal_reachable(self):
        while True:
            self.goal = self.goal_space.sample()
            if np.linalg.norm(self.goal) < 0.2:
                break

    def is_reachable(self):
        return (np.linalg.norm(self.goal) < 0.2)





