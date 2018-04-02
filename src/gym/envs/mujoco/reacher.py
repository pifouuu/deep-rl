import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

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
        qpos[-2:] = np.array([0, 0.1])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        """
        position :
        cos(theta1), cos(theta2), sin(theta1), sin(theta2), theta1', theta2'
        fingertip_x, fingertip_y

        total size : 8
        """
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip")[:2]
        ])

class ReacherOriginEnv(ReacherEnv):
    def __init__(self):
        super(ReacherOriginEnv, self).__init__()

    def _get_obs(self):
        """
        position :
        cos(theta1), cos(theta2), sin(theta1), sin(theta2), theta1', theta2'
        fingertip_x, fingertip_y

        total size : 8
        """
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip")[:2] - self.get_body_com("target")[:2]
        ])


