#!/usr/bin/env bash

SIGMAS=(0.3 0.5 0.8 0.9 1.0 1.3 1.5 2.0 2.5 3.0)

for SIGMA in ${SIGMAS[*]}
do
  export LOGS=logs/first/$SIGMA
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export SIGMA
    export PERF_STUDY="xfirst_$SIGMA"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d . scripts/first-loop.sh
  )
done
