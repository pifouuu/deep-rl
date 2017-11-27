import gym

class CommonConfig:
    def __init__(self):
        self.debug = False

        self.random_seed = True
        self.actor_type = 1
        self.critic_type = 1
        self.data_path = "data/"
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        
        # MCC requires a very big noise, since it's a delayed reward problem
        self.noise_sigma = 3.
        self.batch_size = 128
        self.gamma = .99
        self.target_actor_update = 1e-3

        self.grad_reset = False
        self.actor_reset_threshold = 1e-3

        self.save = False
        self.visualize_train = False
        self.visualize_test = True

        # We have a total of experiment_epochs * epoch_steps steps training steps
        self.epoch_steps = 1000
        self.experiments_nb = 100
        # Number of tests episodes
        self.epoch_tests = 10

        self.env = gym.make('MountainCarContinuous-v0')

        if not self.random_seed:
            self.seed = 10
        else:
            self.seed = None
