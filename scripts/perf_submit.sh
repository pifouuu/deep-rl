#!/usr/bin/env bash

#PBS -l walltime=11:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=5
LOGDIR=./log/

for TRIAL in $(seq $NB_TRIALS)
do
  sleep 10
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    python3.4 src/main.py \
    --log-dir $LOGDIR \
    --max-steps 200000 \
    --save-freq 199000 \
    --memory ${MEMORY} \
    --strategy ${STRAT} \
    --sampler ${SAMPLER} \
    --alpha ${ALPHA} \
    --delta ${DELTA} \
    --sigma ${SIGMA} \
    --activation ${ACTIVATION} \
    --invert-grads ${IVG} \
    --target-clip ${TCLIP} \
    --env ${ENVT} \
    --train-freq 1 \
    --nb-train-iter 1 \
    --eval-freq 1000 \
    --no-render-test \
    > ${LOGS}/${PERF_STUDY}_${TRIAL}.out \
    2> ${LOGS}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
