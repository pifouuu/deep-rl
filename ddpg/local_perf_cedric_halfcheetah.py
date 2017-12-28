from cedric_study import study_ofp
from perf_config_halfcheetah import PerfConfig

config = PerfConfig()

# Experiment attributes

def main_loop():
    tau = 0.05
    name = "simu_Cheetah1_10_3425.rb"
    study_ofp(tau, name, config)


main_loop()
