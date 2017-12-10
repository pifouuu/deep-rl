#!/usr/bin/env bash

#PBS -l walltime=11:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=20
LOGDIR=./results/

for TRIAL in $(seq $NB_TRIALS)
do
  sleep 2
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    python3.4 main.py \
    --summary-dir $LOGDIR \
    --max-steps 200000 \
    --max-episode-steps ${NSTEP} \
    --memory ${MEMORY} \
    --strategy ${STRAT} \
    --sampler ${SAMPLER} \
    --alpha ${ALPHA} \
    --delta ${DELTA} \
    --activation ${ACTIVATION} \
    --invert-grads ${IVG} \
    --target-clip ${TCLIP} \
    > ${LOGS}/${PERF_STUDY}_${TRIAL}.out \
    2> ${LOGS}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
