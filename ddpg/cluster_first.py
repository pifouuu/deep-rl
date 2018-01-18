from trial import trial
from configs.cmc import CMCConfig
import numpy as np
import os

# Configuration
name = os.environ["NAME"]
essai = os.environ["TRIAL"]

config = CMCConfig()
config.study = "first"
config.frozen = False
config.ddpg_noise = True

filepath = "./first/"
filename = filepath + name + str(essai)
nb_steps=-1
while nb_steps<0:
    nb_steps = trial(config)
np.savetxt(filename,{nb_steps})
