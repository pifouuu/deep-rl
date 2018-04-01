from gym import Wrapper
import numpy as np
import math
from gym.spaces import Box
from collections import deque


class no_goal(Wrapper):
    def __init__(self, env, reward_type):
        super(no_goal, self).__init__(env)
        self.goal = None
        self.rec = None
        self.state_to_goal = []
        self.state_to_obs = range(env.observation_space.high.shape[0])
        self.state_to_reached = []
        self.goal_space = Box(np.array([]), np.array([]))
        self.reward_range = [-1, 0]
        self.initial_goal = np.array([])
        self.reward_type = reward_type
        self.epsilon = 0.05

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        goal_reached = agent_state_1[self.state_to_reached]
        goal = self.initial_goal
        vec = goal - goal_reached
        widths = self.goal_space.high - self.goal_space.low
        term = np.linalg.norm(np.divide(vec, widths)) < self.epsilon
        r=0
        if not term:
            if self.reward_type == 'sparse':
                r = -1
            elif self.reward_type == 'dense':
                r = - np.linalg.norm(vec)
        return r, term

    # def eval_exp(self, previous_state_goal, action, state_goal, reward, terminal):
    #     return reward, terminal

    def _reset(self):
        state = self.env.reset()
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()
        return state

    def _step(self,action):
        state, env_reward, env_terminal, info = self.env.step(action)
        reward, terminal = self.eval_exp(self.prev_state, action, state, env_reward,
                                         env_terminal)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()
        return state, reward, terminal, info

    @property
    def state_dim(self):
        return (self.env.observation_space.shape[0],)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)

    @property
    def goal_parameterized(self):
        return False

class goal_basic(Wrapper):
    def __init__(self,env, reward_type):
        super(goal_basic, self).__init__(env)
        self.goal = []
        self.rec = None
        self.state_to_goal = []
        self.state_to_reached = []
        self.state_to_obs = range(env.observation_space.high.shape[0])
        self.goal_space = None
        self.initial_goal = np.array([])
        self.reward_range = [-1, 0]
        self.prev_state = None
        self.starts = deque(maxlen=100)
        self.reward_type = reward_type
        self.epsilon = 0.05

    def add_goal(self, state, goal):
        return np.concatenate([state, goal])

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        goal_reached = agent_state_1[self.state_to_reached]
        goal = agent_state_1[self.state_to_goal]
        vec = goal - goal_reached
        widths = self.goal_space.high - self.goal_space.low
        norm_vec = np.divide(vec, widths)
        term = np.linalg.norm(norm_vec) < self.epsilon
        r=0
        if not term:
            if self.reward_type == 'sparse':
                r = -1
            elif self.reward_type == 'dense':
                r = - np.linalg.norm(norm_vec)
        return r, term

    def set_goal_rnd(self):
        self.goal = self.goal_space.sample()

    def set_goal_reachable(self):
        self.set_goal_rnd()

    def is_reachable(self):
        return True

    def set_goal_init(self):
        self.goal = self.initial_goal

    def change_goal(self, buffer_item, final_state):
        res = buffer_item
        res['state0'][self.state_to_goal] = final_state[self.state_to_reached]
        res['state1'][self.state_to_goal] = final_state[self.state_to_reached]
        res['reward'], res['terminal'] = self.eval_exp(res['state0'],
                                                           res['action'],
                                                           res['state1'],
                                                           res['reward'],
                                                           res['terminal'])
        return res

    def _step(self,action):
        obs, env_reward, env_terminal, info = self.env.step(action)
        state = self.add_goal(obs, self.goal)
        if self.rec is not None: self.rec.capture_frame()
        reward, terminal = self.eval_exp(self.prev_state, action, state, env_reward,
                                         env_terminal)
        self.prev_state = state
        return state, reward, terminal, info

    def _reset(self):
        obs = self.env.reset()
        self.starts.append(obs)
        state = self.add_goal(obs, self.goal)
        if self.rec is not None: self.rec.capture_frame()
        self.prev_state = state
        return state

    def get_start(self):
        start = self.starts[np.random.randint(len(self.starts))]
        return start

    @property
    def state_dim(self):
        return (self.env.observation_space.shape[0]+self.goal_space.shape[0],)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)

    @property
    def goal_parameterized(self):
        return True