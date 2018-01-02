from trial import trial
from configs.cmc import CMCConfig

config = CMCConfig()
config.study = "offline"
config.memory_file = "data/replay_buffer_gep.p"

def main_loop():
    for tau in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:#, 5.0, 10.0, 20.0]:
        config.tau = tau
        for essai in range(0,50):
            config.trial = essai
            trial(config)

main_loop()
