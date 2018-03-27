#!/usr/bin/env bash

#PBS -l walltime=11:00:00
# Request a node per experiment to avoid competition between different TFs
#PBS -l nodes=1:ppn=24
#PBS -V

read -a var1 <<< $PARAM_NAMES
read -a var2 <<< $PARAM_VALS
log=$HOME/deep-rl/log/
command="$HOME/mujoco131env/bin/python3 $HOME/deep-rl/src/main.py "
for i in ${!var1[@]}; do
    param=${var1[$i]}
    val=${var2[$i]}
    command=${command}"--$param $val "
done
command=${command}"--max-steps 1000 --log-dir ${log}"
NB_TRIALS=2
for TRIAL in $(seq $NB_TRIALS)
do
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    $command > ${TEMP_LOG}/${PERF_STUDY}_${TRIAL}.out 2> ${TEMP_LOG}/${PERF_STUDY}_${TRIAL}.err
  ) &
done
wait
