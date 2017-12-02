import os
from ofp_perf_study import perf_study_ofp
from standard_perf_study import perf_study_standard

from perf_config_mcc import PerfConfig

config = PerfConfig()

# Configuration
delta_clip = float(os.environ["DELTA"])
trial = os.environ["TRIAL"]

# Experiment attributes
force = (os.environ.get("FORCE", "false") == "true")

if config.run_type = "ofp":
    perf_study_ofp(delta_clip, trial, config)
elif config.run_type = "standard ":
    perf_study_standard(delta_clip, trial, config)
else:
    print("WTF: unknown run type!")
