import json
from perfcollector_dataframe import PerfCollectorData
import os

directory = "./experiments2/" #"./results/perf_ofp_gep2/"  #

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

def plot_all():
    cpt = 0
    collector = PerfCollectorData("./img/")
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
    collector.plot_all()

plot_all()
