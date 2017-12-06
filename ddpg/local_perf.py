from standard_perf_study import perf_study_standard
from ofp_perf_study import perf_study_ofp
from perf_config_mcc import PerfConfig

config = PerfConfig()

# Experiment attributes

#with replay_buffer_mix, the right instability interval is in delta_clip between 5 and 25. 20 is close to 50% stability
def main_loop():
    #for delta_clip in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 50, 100, 500, 1000]:
    for delta_clip in [1.0]:#, 5.0, 10.0, 20.0]:
        for trial in range(0,100):
            if config.run_type == "ofp":
                perf_study_ofp(delta_clip, trial, config)
            elif config.run_type == "standard":
                perf_study_standard(delta_clip, trial, config)
            else:
                print("WTF: unknown run type!")

main_loop()
