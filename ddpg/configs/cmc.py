import gym
from configs.common import CommonConfig

class CMCConfig(CommonConfig):
    def __init__(self,**kwargs):
        # Basic variables
        super(CMCConfig, self).__init__(**kwargs)

        self.envname = "cmc"
        self.reward_scaling = 1.0
        self.tau = 0.05
        self.delta_clip = 100
        self.eval_episodes = 10 #number of episodes to run during evaluation', default=20
        self.max_episode_steps = 1000
        self.max_steps = 1e7#6000 #
        self.eval_freq = 1000 #evaluation frequency', default=1000
        # CMC requires a big noise, because of a specific problem
        self.noise_factor = 0.3 #3.0
        self.save_step_stats = False
        self.batch_size = 128
        self.averaging = True

        self.env = gym.make('MountainCarContinuous-v0')
        if not self.random_seed:
            self.env.seed(self.seed)

        self.results_root_name = "cmc_results"
        self.memory_file = None