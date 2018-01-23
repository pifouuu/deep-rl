#!/usr/bin/env bash

#PBS -l walltime=01:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=1
LOGDIR=./log/results/
SAVEDIR=./log/saves/

for TRIAL in $(seq $NB_TRIALS)
do
  sleep 2
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    python3.4 src/main.py \
    --summary-dir $LOGDIR \
    --save-dir $SAVEDIR \
    --max-steps 100000 \
    --save-freq 1000000 \
    --memory ${MEMORY} \
    --strategy ${STRAT} \
    --sampler ${SAMPLER} \
    --alpha ${ALPHA} \
    --delta ${DELTA} \
    --sigma ${SIGMA} \
    --activation ${ACTIVATION} \
    --invert-grads ${IVG} \
    --target-clip ${TCLIP} \
    --env ${ENVT}
    > ${LOGS}/${PERF_STUDY}_${TRIAL}.out \
    2> ${LOGS}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
