#!/usr/bin/env bash

TAU=(1e-4 1e-3 1e-2 0.1 1.0 10.0 100.0)
FORCE="true"

for TAU in ${TAU[*]}
do
  export LOGS=logs/perf/$TAU
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export TAU
    export FORCE
    export PERF_STUDY="xperf_$TAU"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d . perf-submit_tau.sh
  )
done
