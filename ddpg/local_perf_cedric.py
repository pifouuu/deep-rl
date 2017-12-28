from cedric_study import study_ofp
from perf_config_mcc import PerfConfig

config = PerfConfig()

# Experiment attributes

def main_loop():
    for i in range(88,137):
        tau = 0.05
        #name = "CMC_buffers/simu_CMC1_"+str(i)+"_buffer"
        #name = "CMC_buffer_tanh/simu_CMC3_"+str(i)+"_buffer"
        name = "CMC_buffer_tanh_20episodes/simu_CMC4_" + str(i) + "_buffer"
        #name = "CMC_buffer_sig/simu_CMC2_" + str(i) + "_buffer"
        study_ofp(tau, name, config)


main_loop()
