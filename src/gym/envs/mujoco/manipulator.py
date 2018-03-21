import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env
import six

class ManipulatorEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, file='manipulator.xml'):
        self.init_env()
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, file, 2)

    def init_env(self):
        self.target = 'target'
        self.objects = []
        self.fixed_objects = []
        self.sites = ['grasp']
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']
        self.arm_bodies = []

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        return ob, 0, False, dict()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        # root_idx = self.model.body_names.index(six.b('upper_arm'))
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
        body_pos = self.model.body_pos.copy()
        body_quat = self.model.body_quat.copy()

        while penetrating:
            # Randomise angles of arm joints.
            indices = [self.model.joint_names.index(six.b(name)) for name in self.arm_joints]
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

            # Randomise object location.
            for object in self.objects:
                object_x = np.random.uniform(-.5, .5)
                object_z = np.random.uniform(0, .7)
                object_angle = np.random.uniform(0, 2 * np.pi)
                qvel[self.model.joint_names.index(six.b(object+'_x'))] = np.random.uniform(-5, 5)
                qpos[self.model.joint_names.index(six.b(object+'_x'))] = object_x
                qpos[self.model.joint_names.index(six.b(object+'_z'))] = object_z
                qpos[self.model.joint_names.index(six.b(object+'_y'))] = object_angle

            # Randomise fixed objects locations
            for fixed in self.fixed_objects:
                fixed_idx = self.model.body_names.index(six.b(fixed))
                fixed_x = np.random.uniform(-.4, .4)
                fixed_z = np.random.uniform(.1, .4)
                fixed_angle = np.random.uniform(-np.pi/3, np.pi/3)
                body_pos[fixed_idx, [0, 2]] = fixed_x, fixed_z
                body_quat[fixed_idx, [0, 2]] = [np.cos(fixed_angle / 2), np.sin(fixed_angle / 2)]

            # Fixed target location ; randomized target in goal wrapper
            target_x = 0.
            target_z = .2
            target_angle = 0.
            target_idx = self.model.body_names.index(six.b(self.target))
            body_pos[target_idx, [0, 2]] = target_x, target_z
            body_quat[target_idx, [0, 2]] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]

            self.model.body_pos = body_pos
            self.model.body_quat = body_quat
            self.set_state(qpos, qvel)

            # Check for collisions.
            # self.render(mode='human')
            penetrating = self.model.data.ncon > 0

        return self._get_obs()

    def touch(self):
        return np.log1p(self.model.data.sensordata)

    def proprioception(self):
        """Returns the arm state, with unbounded angles as sine/cosine."""
        arm = []
        for joint in self.arm_joints:
            idx = self.model.joint_names.index(six.b(joint))
            joint_value = self.model.data.qpos[idx]
            if not self.model.jnt_limited[idx]:
                arm += [np.sin(joint_value), np.cos(joint_value)]
            else:
                arm.append(joint_value)
            # Slightly changed order of attributes
            arm.append(self.model.data.qvel[idx])
        return np.hstack(arm)

    def body_location(self, body):
        """Returns the x,z position and y orientation of a body."""
        body_idx = self.model.body_names.index(six.b(body))
        body_position = self.model.body_pos[body_idx, [0, 2]]
        body_orientation = self.model.body_quat[body_idx, [0, 2]]
        angle = np.arccos(body_orientation[0])
        if np.arcsin(body_orientation[1]) < 0:
            angle = -angle
        return np.hstack((body_position, np.array([2*angle])))


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

        full size : 33
        """
        # Sensors
        proprioception = self.proprioception() # size 9+8 = 17
        touch = self.touch() # size 5 (5 sensors)
        observations = [proprioception, touch]
        # TODO: make sure site_xpos is sufficient to describe the state. site_xmat ?

        # Objects sites for reward computation
        if self.sites:
            site_idx = [self.model.site_names.index(six.b(site)) for site in self.sites]
            site_positions = [self.model.data.site_xpos[idx][[0, 2]] for idx in site_idx]
            sites = np.hstack(site_positions) # size 2*n_sites
            observations.append(sites)

        # Objects body locations for state description
        bodies = self.fixed_objects+self.objects+self.arm_bodies
        if bodies:
            body_locations = [self.body_location(body) for body in self.fixed_objects+self.objects+self.arm_bodies]
            bodies = np.hstack(body_locations) # size 3*(n_fixed_objects+n_objects)
            observations.append(bodies)

        observation_arrays = [observation.ravel() for observation in observations]
        obs = np.concatenate(observation_arrays)
        return obs

class ManipulatorBallEnv(ManipulatorEnv):
    def __init__(self):
        super(ManipulatorBallEnv, self).__init__('manipulator_target_ball.xml')

    def init_env(self):
        self.target = 'target_ball'
        self.objects = ['ball']
        self.fixed_objects = []
        self.sites = []
        self.arm_bodies = []
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']

class ManipulatorBallCupEnv(ManipulatorEnv):
    def __init__(self):

        super(ManipulatorBallCupEnv, self).__init__('manipulator_cup_ball.xml')

    def init_env(self):
        self.target = 'target_ball'
        self.objects = ['ball']
        self.fixed_objects = ['cup']
        self.sites = []
        self.arm_bodies = []
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']

class ManipulatorPegEnv(ManipulatorEnv):
    def __init__(self):

        super(ManipulatorPegEnv, self).__init__('manipulator_target_peg.xml')

    def init_env(self):
        self.target = 'target_peg'
        self.objects = ['peg']
        self.fixed_objects = []
        # self.sites = ['peg_grasp', 'grasp', 'peg_pinch', 'pinch', 'peg', 'target_peg', 'target_peg_tip',
        #                                                  'peg_tip']
        self.sites = []
        self.arm_bodies = []
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']

class ManipulatorPegSlotEnv(ManipulatorEnv):
    def __init__(self):

        super(ManipulatorPegSlotEnv, self).__init__('manipulator_slot_peg.xml')

    def init_env(self):
        self.target = 'target_peg'
        self.objects = ['peg']
        self.fixed_objects = ['slot']
        # self.sites = ['peg_grasp', 'grasp', 'peg_pinch', 'pinch', 'peg', 'target_peg', 'target_peg_tip',
        #                                                  'peg_tip']
        self.sites = []
        self.arm_bodies = []
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']

class ManipulatorBoxesEnv(ManipulatorEnv):
    def __init__(self):

        super(ManipulatorBoxesEnv, self).__init__('manipulator_boxes.xml')

    def init_env(self):
        self.target = 'target_box'
        self.objects = ['box0', 'box1', 'box2', 'box3']
        self.fixed_objects = []
        # self.sites = ['box1', 'box2', 'box3']
        self.sites = []
        self.arm_bodies = []
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']

class PlayroomEnv(ManipulatorEnv):
    def __init__(self):

        super(PlayroomEnv, self).__init__('playroom.xml')

    def init_env(self):
        self.target = 'target_box'
        self.objects = ['ball', 'peg', 'box0', 'box1', 'box2', 'box3']
        self.fixed_objects = ['slot', 'cup']
        # self.sites = ['box1', 'box2', 'box3']
        self.sites = []
        self.arm_bodies = []
        self.arm_joints = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
                           'finger', 'fingertip', 'thumb', 'thumbtip']
