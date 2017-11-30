#!/usr/bin/env bash

#PBS -l walltime=1:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=3
LOGDIR=/home/pierre/PycharmProjects/deep-rl/ddpg/results

for TRIAL in $(seq $NB_TRIALS)
do
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    python3.5 main.py --summary-dir $LOGDIR --wrapper WithGoal --memory SARST --sampler ${SAMPLER} > ${LOGS}/${PERF_STUDY}_${TRIAL}.out 2> ${LOGS}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
