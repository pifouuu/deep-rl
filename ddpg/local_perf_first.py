from first_perf_study import first_perf_study
import matplotlib.pyplot as plt
import numpy as np

# Experiment attributes

def main_loop():
    n_runs = 1000

    n_steps_success = np.zeros([n_runs])

    for i in range(n_runs):
        nb_steps=-1
        while nb_steps<0:
            nb_steps = first_perf_study()
        n_steps_success[i] = nb_steps

    print('mean number of steps before reaching a reward: ', n_steps_success.mean())
    fig = plt.figure()
    plt.hist(n_steps_success, )
    plt.xlabel('number of steps before first reward')
    plt.title('Histogram of the number of steps before reaching the first reward (' + str(n_runs) + ' runs)')


main_loop()
