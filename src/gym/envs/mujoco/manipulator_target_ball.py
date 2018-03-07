import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env
import six


_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
               'finger', 'fingertip', 'thumb', 'thumbtip']
_P_IN_HAND = .1  # Probabillity of object-in-hand initial state
_P_IN_TARGET = .1  # Probabillity of object-in-target initial state

class ManipulatorTargetBallEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'manipulator_target_ball.xml', 2)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, 0, done, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        root_idx = self.model.body_names.index(six.b('upper_arm'))
        # self.viewer.cam.lookat[0] = self.model.body_pos[root_idx, 0]
        # self.viewer.cam.lookat[1] = self.model.body_pos[root_idx, 1]
        # self.viewer.cam.lookat[2] = self.model.body_pos[root_idx, 2]
        self.viewer.cam.distance = 1 * self.model.stat.extent
        self.viewer.cam.lookat[2] = 10
        self.viewer.cam.camid = 0
        # self.viewer.cam.elevation = -1
        self.viewer.cam.fovy = 4

    def reset_model(self):
        """Sets the state of the environment at the start of each episode."""

        # Find a collision-free random initial configuration.
        penetrating = True
        qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        while penetrating:

            # Randomise angles of arm joints.
            indices = [self.model.joint_names.index(six.b(name)) for name in _ARM_JOINTS]
            is_limited = self.model.jnt_limited[indices].astype(np.bool)
            joint_range = self.model.jnt_range[indices]
            lower_range = joint_range[:, 0]
            upper_range = joint_range[:, 1]
            lower_limits = np.where(is_limited.flatten(), lower_range, -np.pi)
            upper_limits = np.where(is_limited.flatten(), upper_range, np.pi)
            angles = np.random.uniform(lower_limits, upper_limits)
            qpos[indices] = angles

            # Symmetrize hand.
            finger_idx = self.model.joint_names.index(six.b('finger'))
            thumb_idx = self.model.joint_names.index(six.b('thumb'))
            qpos[finger_idx] = qpos[thumb_idx]

            # Randomise target location.
            target_x = np.random.uniform(-.4, .4)
            target_z = np.random.uniform(.1, .4)
            target_angle = np.random.uniform(-np.pi, np.pi)
            target_idx = self.model.body_names.index(six.b('target_ball'))
            body_pos = self.model.body_pos.copy()
            body_pos[target_idx, [0, 2]] = target_x, target_z
            self.model.body_pos = body_pos
            body_quat = self.model.body_quat.copy()
            body_quat[target_idx, [0, 2]] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]
            self.model.body_quat = body_quat

            # Randomise object location.
            object_init_probs = [_P_IN_HAND, _P_IN_TARGET, 1 - _P_IN_HAND - _P_IN_TARGET]
            init_type = np.random.choice(['in_hand', 'in_target', 'uniform'], 1,
                                         p=object_init_probs)[0]


            if init_type == 'in_target':
                object_x = target_x
                object_z = target_z
                object_angle = target_angle
            elif init_type == 'in_hand':
                self.model.forward()
                grasp_idx = self.model.site_names.index(six.b('grasp'))
                object_x = self.model.data.site_xpos[grasp_idx, 0]
                object_z = self.model.data.site_xpos[grasp_idx, 2]
                grasp_direction = self.model.data.site_xmat[grasp_idx, [0, 6]]
                object_angle = np.pi - np.arctan2(grasp_direction[1], grasp_direction[0])
            else:

                object_x = np.random.uniform(-.5, .5)
                object_z = np.random.uniform(0, .7)
                object_angle = np.random.uniform(0, 2 * np.pi)
                qvel[self.model.joint_names.index(six.b('ball_x'))] = np.random.uniform(-5, 5)

            qpos[self.model.joint_names.index(six.b('ball_x'))] = object_x
            qpos[self.model.joint_names.index(six.b('ball_z'))] = object_z
            qpos[self.model.joint_names.index(six.b('ball_y'))] = object_angle

            self.set_state(qpos, qvel)
            # Check for collisions.
            penetrating = self.model.data.ncon > 0

        return self._get_obs()

    def touch(self):
        return np.log1p(self.model.data.sensordata)

    def bounded_position(self):
        """Returns the position, with unbounded angles as sine/cosine."""
        state = []
        for joint_id in range(self.model.njnt):
            joint_value = self.model.data.qpos[joint_id]
            if (not self.model.jnt_limited[joint_id] and
                        self.model.jnt_type[joint_id] == 3):  # Unbounded hinge.
                state += [np.sin(joint_value), np.cos(joint_value)]
            else:
                state.append(joint_value)
        return np.asarray(state)

    def body_location(self, body):
        """Returns the x,z position and y orientation of a body."""
        body_idx = self.model.body_names.index(six.b(body))
        body_position = self.model.body_pos[body_idx, [0, 2]]
        body_orientation = self.model.body_quat[body_idx, [0, 2]]
        return np.hstack((body_position, body_orientation))

    def _get_obs(self):
        """
        position:
        sin(armroot), cos(armroot), arm_shoulder, arm_elbow, arm_wrist, thumb, thumb_tip, finger, finger_tip
        ball_x
        ball_z
        sin(ball_y)
        cos(ball_y)

        touch:
        palm, finger, thumb, fingertip, thumbtip

        full size : 37
        """
        position = self.bounded_position() # size 13 (9 limited joints + 2*2 unlimited hinge joints)
        hand = self.body_location('hand') # size 4
        target = self.body_location('target_ball') # size 4
        velocity = self.model.data.qvel[:] # size 11 (11 joints)
        touch = self.touch() # size 5 (5 sensors)
        observations = [position, hand, target, velocity, touch]
        observation_arrays = [observation.ravel() for observation in observations]
        obs = np.concatenate(observation_arrays)
        return obs
