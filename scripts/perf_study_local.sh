#!/usr/bin/env bash

LOGDIR=$HOME/deep-rl/log/local/
TEMPLOG=$HOME/deep-rl/log/local/temp_log/

if [[ $# -gt 2 ]];
then
    SCRIPT=$0
    PARAM="$1"
    read -a PARAM_VALS <<< $2
    for PARAM_VAL in "${PARAM_VALS[@]}"; do
        TEMPLOG=${TEMPLOG}${PARAM}_${PARAM_VAL}
        LOGDIR=${LOGDIR}${PARAM}_${PARAM_VAL}
        param_names="${param_names} ${PARAM}"
        param_values="${param_values} ${PARAM_VAL}"
        shift 2
        $SCRIPT
    done
else
    rm -rf $TEMP_LOG
    mkdir -p $TEMP_LOG
    (
        export TEMP_LOG
        export PERF_STUDY=perf_${ENV}_${PARAM}_${PARAM_VAL}
        export LOGDIR
        rm -f ${PERF_STUDY}.e*
        qsub -N ${PERF_STUDY} -o "$TEMP_LOG/${PERF_STUDY}.out" -b "$TEMP_LOG/${PERF_STUDY}.err" -d $HOME/deep-rl $HOME/PycharmProjects/deep-rl/scripts/perf_submit_local.sh ${param_names} {param_values}
    )
fi