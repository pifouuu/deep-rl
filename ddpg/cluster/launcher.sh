#!/bin/sh
#PBS -N ddpg_mountain_car_standard
#PBS -o ddpg_mountain_car_standard.out
#PBS -b ddpg_mountain_car_standard.err
#PBS -m abe
#PBS -M pierre.fournier@isir.upmc.fr
#PBS -l walltime=01:00:00
#PBS -l ncpus=8
for i in {1..3}
do
    /usr/bin/python3.4 /home/fournier/deep-rl/ddpg/ddpg.py --summary-dir /home/fournier/deep-rl/ddpg/results/ --max-episodes 20 --with-goal --episode-reset
done