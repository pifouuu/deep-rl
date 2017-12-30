from first_perf_study import first_perf_study
import matplotlib.pyplot as plt
import numpy as np

# Experiment attributes

# Configuration
name = os.environ["NAME"]
trial = os.environ["TRIAL"]

filepath = "./first/"
filename = filepath + name + str(trial)
nb_steps=-1
while nb_steps<0:
    nb_steps = first_perf_study()
np.savetxt(filename,{nb_steps})
