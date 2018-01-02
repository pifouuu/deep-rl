import tensorflow as tf
import numpy as np

class CommonConfig:
    def __init__(self):
        self.debug = False

        self.random_seed = False

        if not self.random_seed:
            self.seed = 10
        else:
            self.seed = None

        if not self.random_seed:
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)

        self.actor_type = 1
        self.critic_type = 1
        self.data_path = "data/"
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3

        self.gamma = .99
        self.target_actor_update = 1e-3

        self.grad_reset = False
        self.actor_reset_threshold = 1e-3

        self.save = False
        self.visualize_train = False
        self.visualize_test = True

        # We have a total of experiment_epochs * epoch_steps steps training steps
        self.epoch_steps = 1000

        self.study = None #"from_cedric"  # "standard"  #  "first" # "offline" #  "from_cedric_ofl"#
        self.buffer_name = None
        self.trial = 0

