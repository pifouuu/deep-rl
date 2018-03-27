#!/usr/bin/env bash

param_name=$1
param_vals=$2
if [[ $# -gt 0 ]];
then
    for val in ${param_vals[@]}; do
        (
            shift 2
            export PARAM_NAMES="${PARAM_NAMES} ${param_name}"
            export PARAM_VALS="${PARAM_VALS} ${val}"
            $0 "$@"
        )
    done
fi
if [ $# -eq 0 ];
then
    (
        read -a var1 <<< $PARAM_NAMES
        read -a var2 <<< $PARAM_VALS
        PERF_STUDY=perf
        TEMP_LOG=$HOME/deep-rl/temp_log
        for i in ${!var1[@]}; do
            param=${var1[$i]}
            val=${var2[$i]}
            PERF_STUDY=${PERF_STUDY}_${param}_${val}
            TEMP_LOG=${TEMP_LOG}_${param}_${val}
        done
        rm -rf $TEMP_LOG
        mkdir -p $TEMP_LOG
        rm -f ${PERF_STUDY}.e*
        export PARAM_NAMES
        export PARAM_VALS
        export TEMP_LOG
        export PERF_STUDY
        qsub -N ${PERF_STUDY} -o "$TEMP_LOG/${PERF_STUDY}.out" -b "$TEMP_LOG/${PERF_STUDY}.err" -d $HOME/deep-rl $HOME/deep-rl/scripts/test2_cluster.sh
    )
fi