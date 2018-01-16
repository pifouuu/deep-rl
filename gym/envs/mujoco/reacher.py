import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env


# class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         utils.EzPickle.__init__(self)
#         mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
#
#     def _step(self, a):
#         vec = self.get_body_com("fingertip")-self.get_body_com("target")
#         reward_dist = - np.linalg.norm(vec)
#         reward_ctrl = - np.square(a).sum()
#         reward = reward_dist + reward_ctrl
#         self.do_simulation(a, self.frame_skip)
#         ob = self._get_obs()
#         done = False
#         return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
#
#     def viewer_setup(self):
#         self.viewer.cam.trackbodyid = 0
#
#     def reset_model(self):
#         qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
#         while True:
#             self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
#             if np.linalg.norm(self.goal) < 2:
#                 break
#         qpos[-2:] = self.goal
#         qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
#         qvel[-2:] = 0
#         self.set_state(qpos, qvel)
#         return self._get_obs()
#
#     def _get_obs(self):
#         theta = self.model.data.qpos.flat[:2]
#         return np.concatenate([
#             np.cos(theta),
#             np.sin(theta),
#             self.model.data.qpos.flat[2:],
#             self.model.data.qvel.flat[:2],
#             self.get_body_com("fingertip") - self.get_body_com("target")
#         ])

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        self.action_bounds = self.action_space.high
        self.min_reward = -2.4
        self.max_reward = 0
        self.goal_dim = 0


    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # self.goal = self.get_random_goal()
        self.goal = np.array([0, 0.1])
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        theta_dot = self.model.data.qvel.flat[:2]
        fingertip_com = self.get_body_com("fingertip")
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            theta_dot,
            fingertip_com,
        ])

    def eval_exp(self, _, action, agent_state_1):
        pass

    def get_random_goal(self):
        return []

    def get_initial_goal(self):
        return []



    @property
    def has_goal(self):
        return False



