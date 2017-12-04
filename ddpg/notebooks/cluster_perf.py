from fbddpg_perf import perf_study
from hooks.perfhook import PerfHook
from rl.runtime.experiment import Experiment
import os

# Configuration
delta_clip = float(os.environ["DELTA"])
trial = os.environ["TRIAL"]

# Experiment attributes
force = (os.environ.get("FORCE", "false") == "true")

experiment = Experiment("perf/{}/{}".format(delta_clip, trial), force=True, hooks=[PerfHook()])

perf_study(experiment, delta_clip)
