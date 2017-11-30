import json
import numpy as np
from notebooks.perfcollector_common import PerfCollectorCommon
from notebooks.perfcollector import PerfCollector
import os

directory = "./perf_ofp/"  # "./experiments/"

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


def plot_common():
    cpt = 0
    collector = PerfCollectorCommon("./img/")
    collector.init()
    for delta in os.listdir(directory):
        perf_values = {}
        experiment_path = directory + delta + "/"
        for file in os.listdir(experiment_path):
            filename = experiment_path + file + "/log_episodes/progress.json"
            perf_values = get_perf_values(filename)
            cpt += 1
            collector.add(perf_values)
    collector.plot()
    print(cpt, " files found")
    collector.stats()

def plot_by_delta():
    cpt = 0
    collector = PerfCollector("./img/")
    for delta in os.listdir(directory):
        collector.init(delta)
        perf_values = {}
        experiment_path = directory + delta + "/"
        for file in os.listdir(experiment_path):
            filename = experiment_path + file + "/log_episodes/progress.json"
            perf_values = get_perf_values(filename)
            cpt += 1
            collector.add(delta, perf_values)
        collector.plot(delta)
    print(cpt, " files found")
    collector.stats()

plot_common()
plot_by_delta()