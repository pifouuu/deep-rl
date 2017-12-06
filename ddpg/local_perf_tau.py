from tau_study import tau_study_standard
from perf_config_mcc import PerfConfig

config = PerfConfig()

# Experiment attributes

def main_loop():
    for tau in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:#, 5.0, 10.0, 20.0]:
        for trial in range(0,50):
            if config.run_type == "ofp":
                tau_study_standard(tau, trial, config)
            elif config.run_type == "standard":
                tau_study_standard(tau, trial, config)
            else:
                print("WTF: unknown run type!")

main_loop()
