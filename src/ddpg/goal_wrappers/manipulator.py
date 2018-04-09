from .wrapper import goal_basic, no_goal
import numpy as np
import six


from gym.spaces import Box

class BaseNoGoal(no_goal):
    def __init__(self, env, reward_type, epsilon):
        super(BaseNoGoal, self).__init__(env, reward_type, epsilon)
        self.state_to_reached = range(22, 24)
        self.target = 'target'
        self.initial_goal = np.array([0., 0.2])


    def _reset(self):
        _ = self.env.reset()

        # Randomise target location
        # TODO: integrate possibility to have target in receptacle more easily
        target_x = self.initial_goal[0]
        target_z = self.initial_goal[1]

        target_idx = self.unwrapped.model.body_names.index(six.b(self.target))
        body_pos = self.unwrapped.model.body_pos.copy()
        body_pos[target_idx, [0, 2]] = target_x, target_z

        self.unwrapped.model.body_pos = body_pos

        state = self.unwrapped._get_obs()
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()
        return state

class Base(goal_basic):
    def __init__(self, env, reward_type, epsilon):
        super(Base, self).__init__(env, reward_type, epsilon)
        self.state_to_goal = range(24,26)
        self.state_to_obs = range(24)
        self.state_to_reached = range(22,24)
        self.goal_space = Box(np.array([-.4, .1]), np.array([.4, .4]))
        self.target = 'target'
        self.initial_goal = np.array([0., 0.2])


    def _reset(self):
        _ = self.env.reset()

        # Randomise target location
        # TODO: integrate possibility to have target in receptacle more easily
        target_x = self.goal[0]
        target_z = self.goal[1]

        target_idx = self.unwrapped.model.body_names.index(six.b(self.target))
        body_pos = self.unwrapped.model.body_pos.copy()
        body_pos[target_idx, [0, 2]] = target_x, target_z

        self.unwrapped.model.body_pos = body_pos

        obs = self.unwrapped._get_obs()
        self.starts.append(obs)
        state = self.add_goal(obs, self.goal)
        self.prev_state = state
        return state

class Ball(Base):
    def __init__(self, env, reward_type, epsilon):
        super(Ball, self).__init__(env, reward_type, epsilon)
        self.target = 'target_ball'
        self.state_to_goal = range(25, 27)
        self.state_to_obs = range(25)
        self.state_to_reached = range(22, 24)

class BallNoGoal(BaseNoGoal):
    def __init__(self, env, reward_type, epsilon):
        super(BallNoGoal, self).__init__(env, reward_type, epsilon)
        self.target = 'target_ball'
        self.state_to_reached = range(22, 24)


class BallCup(Base):
    def __init__(self, env, reward_type, epsilon):
        super(BallCup, self).__init__(env, reward_type, epsilon)
        self.target = 'target_ball'
        self.state_to_goal = range(28, 30)
        self.state_to_obs = range(28)
        self.state_to_reached = range(25, 27)

class BallCupNoGoal(BaseNoGoal):
    def __init__(self, env, reward_type, epsilon):
        super(BallCupNoGoal, self).__init__(env, reward_type, epsilon)
        self.target = 'target_ball'
        self.state_to_reached = range(25, 27)

class Peg(Base):
    def __init__(self, env, reward_type, epsilon):
        super(Peg, self).__init__(env, reward_type, epsilon)
        self.target = 'target_peg'
        self.state_to_goal = range(25, 28)
        self.state_to_obs = range(25)
        self.state_to_reached = range(22, 25)
        self.goal_space = Box(np.array([-.4, .1, -np.pi/3]), np.array([.4, .4, np.pi/3]))
        self.initial_goal = np.array([0., 0.2, 0])

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

        obs = self.unwrapped._get_obs()
        self.starts.append(obs)
        state = self.add_goal(obs, self.goal)
        self.prev_state = state
        return state

class PegNoGoal(BaseNoGoal):
    def __init__(self, env, reward_type, epsilon):
        super(PegNoGoal, self).__init__(env, reward_type, epsilon)
        self.target = 'target_peg'
        self.state_to_reached = range(22, 25)
        self.initial_goal = np.array([0., 0.2, 0])

    def _reset(self):
        _ = self.env.reset()

        # Randomise target location
        # TODO: integrate possibility to have target in receptacle more easily
        target_x = self.initial_goal[0]
        target_z = self.initial_goal[1]
        target_angle = self.initial_goal[2]

        target_idx = self.unwrapped.model.body_names.index(six.b(self.target))
        body_pos = self.unwrapped.model.body_pos.copy()
        body_pos[target_idx, [0, 2]] = target_x, target_z
        body_quat = self.unwrapped.model.body_quat.copy()
        body_quat[target_idx, [0, 2]] = [np.cos(target_angle / 2), np.sin(target_angle / 2)]

        self.unwrapped.model.body_quat = body_quat
        self.unwrapped.model.body_pos = body_pos

        state = self.unwrapped._get_obs()
        self.prev_state = state
        return state


class PegSlot(Peg):
    def __init__(self, env, reward_type, epsilon):
        super(PegSlot, self).__init__(env, reward_type, epsilon)
        self.target = 'target_peg'
        self.state_to_goal = range(28, 31)
        self.state_to_obs = range(28)
        self.state_to_reached = range(25, 28)

class PegSlotNoGoal(PegNoGoal):
    def __init__(self, env, reward_type, epsilon):
        super(PegSlotNoGoal, self).__init__(env, reward_type, epsilon)
        self.target = 'target_peg'
        self.state_to_reached = range(25, 28)

class Boxes(Base):
    def __init__(self, env, reward_type, epsilon):
        super(Boxes, self).__init__(env, reward_type, epsilon)
        self.target = 'target_box'
        self.state_to_goal = range(34, 36)
        self.state_to_obs = range(34)
        self.state_to_reached = range(22, 34)

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = -1
        goal_reached = agent_state_1[self.state_to_reached]
        goal = agent_state_1[self.state_to_goal]
        term = False
        for i in range(4):
            vec = goal - goal_reached[[3*i, 3*i+1]]
            term = term or np.linalg.norm(vec) < 0.1
        if term:
            r = 0
        return r, term

class BoxesNoGoal(BaseNoGoal):
    def __init__(self, env, reward_type, epsilon):
        super(BoxesNoGoal, self).__init__(env, reward_type, epsilon)
        self.target = 'target_box'
        self.state_to_reached = range(22, 34)

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        r = -1
        goal_reached = agent_state_1[self.state_to_reached]
        goal = self.initial_goal
        term = False
        for i in range(4):
            vec = goal - goal_reached[[3*i, 3*i+1]]
            term = term or np.linalg.norm(vec) < 0.1
        if term:
            r =0
        return r, term

class Playroom(Base):
    #TODO
    def __init__(self, env, reward_type, epsilon):
        super(Playroom, self).__init__(env, reward_type, epsilon)
        self.target = 'target_box'
        self.state_to_goal = range(28, 30)
        self.state_to_obs = range(46)
        self.state_to_reached = range(25, 27)

class PlayroomNoGoal(BaseNoGoal):
    def __init__(self, env, reward_type, epsilon):
        super(PlayroomNoGoal, self).__init__(env, reward_type, epsilon)
        self.target = 'target_ball'
        self.state_to_reached = range(28, 30)