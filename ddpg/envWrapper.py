import numpy as np
import math

class NoGoal(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.state_shape = (2,)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.obs_to_goal = [0]
        self.eps = 0.1


    def process_observation(self, observation, goal):
        return observation

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if state1[self.obs_to_goal] >= 0.45:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def evaluate_goal(self, state):
        return True

    def sample_initial_goal(self):
        return [0.45]

    def sample_goal(self):
        return self.sample_initial_goal()

    def sample_random_goal(self, obs):
        goal_found = False
        while not goal_found:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            goal_found = np.abs(obs[self.obs_to_goal]-goal) > self.eps
        return goal

    def process_step(self, state0, goal, action, new_obs, r_env, done_env, info):
        # Compute next complete state
        state1 = self.process_observation(new_obs, goal)

        # Compute reward and terminal condition
        r, done = self.evaluate_transition(state0, action, state1)

        buffer_item = {'state0': state0,
                       'action': action,
                       'reward': r,
                       'state1': state1,
                       'terminal1': done}

        return buffer_item

class RandomGoal(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.obs_to_goal = [0]
        self.state_to_obs = [0,1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.eps = 0.1
        self.has_goal = True

    def process_observation(self, observation, goal):
        return np.concatenate([observation,goal])

    def process_step(self, state0, goal, action, new_obs, r_env, done_env, info):
        # Compute next complete state
        state1 = self.process_observation(new_obs, goal)

        # Compute reward and terminal condition
        r, done = self.evaluate_transition(state0, action, state1)

        buffer_item = {'state0': state0,
                       'action': action,
                       'reward': r,
                       'state1': state1,
                       'terminal1': done}

        return buffer_item

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if np.abs(state1[self.state_to_obs][self.obs_to_goal] - state1[self.state_to_goal]) < self.eps:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self, obs, successes):
        goal_found = False
        while not goal_found:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            goal_found = np.abs(obs[self.obs_to_goal]-goal) > self.eps
        return goal

    def sample_initial_goal(self):
        return [0.45]

class InitialGoal(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.obs_to_goal = [0]
        self.state_to_obs = [0,1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.eps = 0.1
        self.has_goal = True

    def process_observation(self, observation, goal):
        return np.concatenate([observation,goal])

    def process_step(self, state0, goal, action, new_obs, r_env, done_env, info):
        # Compute next complete state
        state1 = self.process_observation(new_obs, goal)

        # Compute reward and terminal condition
        r, done = self.evaluate_transition(state0, action, state1)

        buffer_item = {'state0': state0,
                       'action': action,
                       'reward': r,
                       'state1': state1,
                       'terminal1': done}

        return buffer_item

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if np.abs(state1[self.state_to_obs][self.obs_to_goal] - state1[self.state_to_goal]) < self.eps:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self, obs, successes):
        return [0.45]

    def sample_initial_goal(self):
        return [0.45]

class HandmadeCurriculum(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.obs_to_goal = [0]
        self.state_to_obs = [0, 1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.eps = 0.2
        self.disc_goal_space = np.arange(-1.2, 0.7, 0.1)
        self.difficulties = np.array([6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.means = np.linspace(4, 10, 100)
        self.stds = np.linspace(0.5, 2, 100)

    def process_observation(self, observation, goal):
        return np.concatenate([observation,goal])

    def process_step(self, state0, goal, action, new_obs, r_env, done_env, info):
        # Compute next complete state
        state1 = self.process_observation(new_obs, goal)

        # Compute reward and terminal condition
        r, done = self.evaluate_transition(state0, action, state1)

        buffer_item = {'state0': state0,
                       'action': action,
                       'reward': r,
                       'state1': state1,
                       'terminal1': done}

        return buffer_item

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if np.abs(state1[self.state_to_obs][self.obs_to_goal] - state1[self.state_to_goal]) < self.eps:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self, obs, successes):
        mean = self.means[min(successes, 99)]
        std = self.stds[min(successes, 99)]
        difficulty = np.clip(int(np.random.normal(mean,std)), 1, 10)
        indices = np.squeeze(np.argwhere(self.difficulties==difficulty), axis=1)
        idx = np.random.choice(indices)
        goal = np.random.uniform(self.disc_goal_space[idx], self.disc_goal_space[idx+1], (1,))
        return difficulty, goal

    def sample_initial_goal(self):
        return [0.45]

    def sample_random_goal(self, obs):
        goal_found = False
        while not goal_found:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            goal_found = np.abs(obs[self.obs_to_goal]-goal) > self.eps
        return goal

class Curriculum(object):
    def __init__(self):
        # Specific to continuous mountain car
        self.obs_to_goal = [0]
        self.state_to_obs = [0, 1]
        self.state_to_goal = [2]
        self.state_shape = (3,)
        self.action_shape = (1,)
        self.reward_shape = (1,)
        self.terminal_shape = (1,)
        self.eps = 0.05
        self.disc_goal_space = np.arange(-1.2, 0.7, 0.1)
        self.difficulties = [6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.means = np.linspace(2, 10, 100)
        self.stds = np.linspace(0.5, 9, 100)

    def process_observation(self, observation, goal):
        return np.concatenate([observation,goal])

    def process_step(self, state0, goal, action, new_obs, r_env, done_env, info):
        # Compute next complete state
        state1 = self.process_observation(new_obs, goal)

        # Compute reward and terminal condition
        r, done = self.evaluate_transition(state0, action, state1)

        buffer_item = {'state0': state0,
                       'action': action,
                       'reward': r,
                       'state1': state1,
                       'terminal1': done}

        return buffer_item

    def evaluate_transition(self, state0, action, state1):
        r = 0
        term = False
        if np.abs(state1[self.state_to_obs][self.obs_to_goal] - state1[self.state_to_goal]) < self.eps:
            r += 100
            term = True
        r -= math.pow(action[0], 2) * 0.1
        return r, term

    def sample_goal(self, obs, successes):
        mean = self.means[min(successes, 99)]
        std = self.stds[min(successes, 99)]
        difficulty = np.clip(int(np.random.normal(mean,std)), 1, 10)
        indices = np.squeeze(np.argwhere(self.difficulties==difficulty))
        idx = np.random.choice(indices)
        goal = np.random.uniform(self.disc_goal_space[idx], self.disc_goal_space[idx+1], (1,))
        return difficulty, goal

    def sample_initial_goal(self):
        return [0.45]

    def sample_random_goal(self, obs):
        goal_found = False
        while not goal_found:
            goal = np.random.uniform([-1.2], [0.6], (1,))
            goal_found = np.abs(obs[self.obs_to_goal]-goal) > self.eps
        return goal
