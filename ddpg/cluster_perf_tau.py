import os
from tau_study import tau_study_standard

from perf_config_mcc import PerfConfig

config = PerfConfig()

# Configuration
tau = float(os.environ["TAU"])
trial = os.environ["TRIAL"]

# Experiment attributes
force = (os.environ.get("FORCE", "false") == "true")

if config.run_type == "ofp":
    tau_study_standard(tau, trial, config)
elif config.run_type == "standard":
    tau_study_standard(tau, trial, config)
else:
    print("WTF: unknown run type!")