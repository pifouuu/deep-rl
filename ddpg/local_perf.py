from trial import trial
from configs.cmc import CMCConfig

config = CMCConfig()

config.study = "standard"
config.ddpg_noise = True
config.visu_policy = True

# Experiment attributes

#with replay_buffer_mix, the right instability interval is in delta_clip between 5 and 25. 20 is close to 50% stability
def main_loop():
    #for delta_clip in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 20, 50, 100, 500, 1000]:
    for delta_clip in [1.0]:#, 5.0, 10.0, 20.0]:
        config.delta_clip = delta_clip
        for essai in range(0,100):
            config.trial = essai
            trial(config)

main_loop()
