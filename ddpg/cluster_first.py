from trial import trial
from configs.cmc import CMCConfig
import numpy as np
import os

# Configuration
#sigma = os.environ["SIGMA"]
essai = os.environ["TRIAL"]
name  = os.environ["NAME"]

config = CMCConfig()
config.study = "first"
config.frozen = False
config.ddpg_noise = True
#config.noise_factor = float(sigma)

filepath = "./first/" + name + "/" #+ sigma + "/"
os.makedirs(filepath, exist_ok=True)
filename = filepath + str(essai) + ".txt"
file = open(filename,"w")
nb_steps=-1
while nb_steps<0:
    nb_steps = trial(config)

file.write(str(nb_steps))
