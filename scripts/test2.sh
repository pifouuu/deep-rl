#!/usr/bin/env bash

read -a var1 <<< $PARAM_NAMES
read -a var2 <<< $PARAM_VALS
log=./

command="$HOME/PycharmProjects/mujoco131env/bin/python3 $HOME/PycharmProjects/deep-rl/src/main.py "
for i in ${!var1[@]}; do
    param=${var1[$i]}
    val=${var2[$i]}
    echo $param
    echo $val
    log=${log}_${param}_${val}
    command=${command}"--$param $val "
done


command=${command}"--max-steps 1000 "
echo $command
rm -rf $log
mkdir -p $log
NB_TRIALS=2
for TRIAL in $(seq $NB_TRIALS)
do
  (
    echo "Running experiment $TRIAL"
    export TRIAL
    $command > ${log}/test_${TRIAL}.out 2> ${log}/test_${TRIAL}.err
  ) &
done
wait
