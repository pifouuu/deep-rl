import os
from trial import trial
from configs.cmc import CMCConfig
from configs.halfcheetah import HCConfig


type = "cmc"
#type = "halfcheetah"

if type=="cmc":
    config = CMCConfig()
elif type=="halfcheetah":
    config = HCConfig()

config.study = "from_cedric"
config.frozen = False
config.trial = os.environ["TRIAL"]
config.name = os.environ["NAME"]

def main_loop():
    if type == "cmc":
        '''
        for i in range(88,137):
            #name = "CMC_buffers/simu_CMC1_"+str(i)+"_buffer"
            #name = "CMC_buffer_tanh/simu_CMC3_"+str(i)+"_buffer"
            config.buffer_name = "cedric_buffers/CMC_buffer_tanh_20episodes/simu_CMC4_" + str(i) + "_buffer"
            #name = "CMC_buffer_sig/simu_CMC2_" + str(i) + "_buffer"
            trial(config)
        '''
        config.buffer_name = "simu_CMC4_88_buffer"
        trial(config)
    elif type == "halfcheetah":
        #config.buffer_name = "cedric_buffers/simu_Cheetah1_14_buffer_1500k_score1700.txt"
        config.buffer_name = "simu_Cheetah1_13_buffer_50k_score1432.txt"
        trial(config)



main_loop()

