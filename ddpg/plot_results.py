import json
import numpy as np
from notebooks.perfcollector_common import PerfCollector
import os


def get_perf_values(filename):
    with open(filename, 'r') as json_data:

        lines = json_data.readlines()
        eval_rewards = []
        for line in lines:

            episode_data = json.loads(line)
            if 'New Test reward' in episode_data:
                step = episode_data['New Training steps']
                perf = episode_data['New Test reward']
                tmp = [step, perf]
                eval_rewards.append(tmp)
    return eval_rewards


experiment_root = "../perf_ofp/"  # "../experiments/"
cpt = 0
collec = PerfCollector()
collec.init()
for delta in os.listdir(experiment_root):
    perf_values = {}
    experiment_path = experiment_root + delta + "/"
    for file in os.listdir(experiment_path):
        filename = experiment_path + file + "/progress.json"  # log_episodes/
        perf_values = get_perf_values(filename)
        cpt += 1
        collec.add(perf_values)
collec.plot()

print(cpt, " files found")
collec.stats()