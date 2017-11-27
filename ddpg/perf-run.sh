#!/usr/bin/env bash

DELTAS=(1 5 10 15 20 50 100 0.5 0.1)
FORCE="true"

for DELTA in ${DELTAS[*]}
do
  export LOGS=logs/perf/$DELTA
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export DELTA
    export FORCE
    export PERF_STUDY="xperf_$DELTA"
    rm -f $LOGS/${PERF_STUDY}.e* $LOGS/${PERF_STUDY}.o* ${PERF_STUDY}.e* ${PERF_STUDY}.o*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d $HOME/RLAgents/sigaud/code perf-submit.sh
  )
done
