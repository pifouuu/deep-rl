from ofp_perf_study import perf_study
#from standard_perf_study import perf_study
import os

# Configuration
delta_clip = float(os.environ["DELTA"])
trial = os.environ["TRIAL"]

# Experiment attributes
force = (os.environ.get("FORCE", "false") == "true")

perf_study(delta_clip,trial)
