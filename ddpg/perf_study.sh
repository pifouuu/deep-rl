#!/usr/bin/env bash

SAMPLERS=(rnd init)

for SAMPLER in ${SAMPLERS[*]}
do
  export LOGS=logs/perf/$SAMPLER
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export PERF_STUDY="perf_$SAMPLER"
    rm -f ${PERF_STUDY}.e*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d $HOME/deep-rl/ddpg perf_submit.sh
  )
done