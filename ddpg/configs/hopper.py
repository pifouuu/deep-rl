import gym
from configs.common import CommonConfig

class HopConfig(CommonConfig):
    def __init__(self,**kwargs):
        # Basic variables
        super(HopConfig, self).__init__(**kwargs)

        #self.memory_file = "data/replay_buffer_us_frequent.p"
        #self.memory_file = "data/replay_buffer_us_fair_big.p"
        #self.memory_file = "data/replay_buffer_gep.p"
        self.envname = "hopper"
        self.reward_scaling = 1.0
        self.tau = 1e-3
        self.eval_episodes = 10 #number of episodes to run during evaluation', default=20
        self.max_episode_steps = 1000
        self.max_steps = 1e7
        self.eval_freq = 1000 #evaluation frequency', default=1000
        self.noise_factor = 0.3 #3.0
        self.save_step_stats = False
        self.batch_size = 128
        self.averaging = True

        self.env = gym.make('Hopper-v1')
        if not self.random_seed:
            self.env.seed(self.seed)

        self.results_root_name = "hop_results"