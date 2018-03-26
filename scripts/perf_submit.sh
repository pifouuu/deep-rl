#!/usr/bin/env bash

#PBS -l walltime=11:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V
NB_TRIALS=5
for TRIAL in $(seq $NB_TRIALS)
do
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    $HOME/mujoco131env/bin/python3 $HOME/deep-rl/src/main.py \
    --env ${ENV} \
    --${PARAM} ${PARAM_VAL} \
    --log-dir $LOGDIR \
    > ${TEMP_LOG}/${PERF_STUDY}_${TRIAL}.out \
    2> ${TEMP_LOG}/${PERF_STUDY}_${TRIAL}.err
  ) &
done

wait
