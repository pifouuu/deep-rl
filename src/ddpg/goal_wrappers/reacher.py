from .wrapper import goal_basic
import numpy as np

from mujoco_py.mjlib import mjlib

class ReacherBenchmark(goal_basic):
    def __init__(self, env):
        super(ReacherBenchmark, self).__init__(env)
        self.goal = None
        self.goals = []
        self.state_to_goal = []
        self.state_to_obs = [0, 1, 2, 3, 6, 7, 8, 9, 10]
        self.obs_to_goal = []
        self.start = np.array([0.2, 0])
        self.initial_goal = np.array([0, 0.1])
        self.reward_range = [-2.4, 0]
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

    def add_goal(self, state, goal):
        return state

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        vec = agent_state_1[[8,9,10]]
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        r = reward_dist + reward_ctrl
        term = False
        return r, term

    def get_random_goal(self):
        while True:
            goal = np.random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(goal) < .2 and np.linalg.norm(goal-self.start) > 0.05:
                break
        return goal

    def change_goal(self, buffer_item, final_state):
        res = buffer_item
        res['state0'][[8, 9, 10]] = 0
        res['state0'][[4, 5]] = final_state[[8, 9]] + final_state[[4, 5]]
        res['state1'][[8, 9, 10]] = 0
        res['state1'][[4, 5]] = final_state[[8, 9]] + final_state[[4, 5]]
        res['reward'], res['terminal'] = self.eval_exp(res['state0'],
                                                           res['action'],
                                                           res['state1'],
                                                           res['reward'],
                                                           res['terminal'])
        return res


class ReacherSparse(ReacherBenchmark):
    def __init__(self, env):
        super(ReacherSparse, self).__init__(env)
        self.reward_range = [-0.2, 100]

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = 0
        vec = agent_state_1[[8, 9, 10]]
        term = np.linalg.norm(vec) < 0.05
        if term:
            r += 100
        r -= 0.1 * np.square(action).sum()
        return r, term





