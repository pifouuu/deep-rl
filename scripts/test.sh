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
        export PARAM_NAMES
        export PARAM_VALS
        bash ./test2.sh
    )
fi