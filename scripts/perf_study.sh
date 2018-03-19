#!/usr/bin/env bash

ENV="$1"
PARAM="$2"
read -a PARAM_VALS <<< $3

LOGDIR=$HOME/deep-rl/log

for PARAM_VAL in ${PARAM_VALS[@]}; do
    export TEMP_LOG=$LOGDIR/temp_log_${ENV}_${PARAM}_${PARAM_VAL}
    rm -rf $TEMP_LOG
    mkdir -p $TEMP_LOG
    (
        export TEMP_LOG
        export ENV
        export PARAM
        export PARAM_VAL
        export PERF_STUDY=perf_${ENV}_${PARAM}_${PARAM_VAL}
        export LOGDIR
        rm -f ${PERF_STUDY}.e*
        qsub -N ${PERF_STUDY} -o "$TEMP_LOG/${PERF_STUDY}.out" -b "$TEMP_LOG/${PERF_STUDY}.err" -d $HOME/deep-rl $HOME/deep-rl/scripts/perf_submit.sh
    )
done