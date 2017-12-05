#!/usr/bin/env bash

#PBS -l walltime=11:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=3
LOGDIR=/home/fournier/deep-rl/ddpg/results/

for TRIAL in $(seq $NB_TRIALS)
do
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    python3.4 main.py \
    --summary-dir $LOGDIR \
    --memory sarst \
    --strategy final \
    --sampler ${SAMPLER} \
    --alpha 1 \
    --delta inf \
    --activation tanh\
    > ${LOGS}/${PERF_STUDY}_${TRIAL}.out \
    2> ${LOGS}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
