from trial import trial
from configs.cmc import CMCConfig
from configs.halfcheetah import HCConfig
from configs.hopper import HopConfig

#envname = "cmc"
#envname = "halfcheetah"
envname = "hopper"

if envname== "cmc":
    config = CMCConfig()
elif envname== "halfcheetah":
    config = HCConfig()
elif envname == "hopper":
    config = HopConfig()

config.study = "standard"# "from_cedric"
config.run_type = "standard"# "ofp" #not used any more in trial() => some refactoring would help
config.frozen = False

def main_loop():
    if envname == "cmc":
        for i in range(88,137):
            #name = "CMC_buffers/simu_CMC1_"+str(i)+"_buffer"
            #name = "CMC_buffer_tanh/simu_CMC3_"+str(i)+"_buffer"
            config.buffer_name = "cedric_buffers/CMC_buffer_tanh_20episodes/simu_CMC4_" + str(i) + "_buffer"
            #name = "CMC_buffer_sig/simu_CMC2_" + str(i) + "_buffer"
            trial(config)
    elif envname == "halfcheetah":
        #config.buffer_name = "cedric_buffers/simu_Cheetah1_14_buffer_1500k_score1700.txt"
        config.buffer_name = "cedric_buffers/simu_Cheetah1_13_buffer_50k_score1432.txt"
        trial(config)
    elif envname == "hopper":
        # config.buffer_name = "cedric_buffers/simu_Cheetah1_14_buffer_1500k_score1700.txt"
        config.buffer_name = "cedric_buffers/simu_Cheetah1_13_buffer_50k_score1432.txt"
        trial(config)
    else:
        print("environment unknown")



main_loop()
