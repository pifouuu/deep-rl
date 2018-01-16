import os
from trial import trial

from configs.cmc import CMCConfig

# Configuration
config = CMCConfig()
config.study = "offline"
config.memory_file = "data/replay_buffer_gep.p"

run_type = "tau"

config.trial = os.environ["TRIAL"]
if run_type == "delta":
    config.delta_clip = float(os.environ["DELTA"])
elif run_type == "tau":
    config.tau = float(os.environ["TAU"])

trial(config)

