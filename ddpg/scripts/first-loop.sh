#!/usr/bin/env bash

#PBS -l walltime=00:59:59
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NAMES=(a b c d e f g h i j k l m n o p q r s t)
NB_TRIALS=50

for NAME in ${NAMES[*]}
do
  (
    export NAME
    for TRIAL in $(seq $NB_TRIALS)
    do(
        echo "Running experiment $NAME $TRIAL"
        export TRIAL
        python cluster_first.py > ${LOGS}/${PERF_STUDY}_${NAME}_${TRIAL}.out 2> ${LOGS}/${PERF_STUDY}_${NAME}_${TRIAL}.err
    ) &
    done
  ) &
  done

wait
