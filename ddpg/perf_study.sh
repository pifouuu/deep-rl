#!/usr/bin/env bash

MEMORIES=(sarst)
STRATS=(final)
SAMPLERS=(rnd)
ALPHAS=(1)
DELTAS=(inf)
ACTIVATIONS=(tanh)
IVGS=(False)
TCLIPS=(False)

for MEMORY in ${MEMORIES[*]}
do
    for STRAT in ${STRATS[*]}
    do
        for SAMPLER in ${SAMPLERS[*]}
        do
            for ALPHA in ${ALPHAS[*]}
            do
                for DELTA in ${DELTAS[*]}
                do
                    for ACTIVATION in ${ACTIVATIONS[*]}
                    do
                        for IVG in ${IVGS[*]}
                        do
                            for TCLIP in ${TCLIPS[*]}
                            do
                                export LOGS=logs/perf/$MEMORY_$STRAT_$SAMPLER_$ALPHA_$DELTA_$ACTIVATION_$IVG_$TCLIP
                                rm -rf $LOGS
                                mkdir -p $LOGS
                                (
                                    export LOGS
                                    export MEMORY
                                    export STRAT
                                    export SAMPLER
                                    export ALPHA
                                    export DELTA
                                    export ACTIVATION
                                    export IVG
                                    export TCLIP
                                    export PERF_STUDY="perf_$MEMORY_$STRAT_$SAMPLER_$ALPHA_$DELTA_$ACTIVATION_$IVG_$TCLIP"
                                    rm -f ${PERF_STUDY}.e*
                                    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d $HOME/deep-rl/ddpg perf_submit.sh
                                )
                            done
                        done
                    done
                done
            done
        done
    done
done


for SAMPLER in ${SAMPLERS[*]}
do
  export LOGS=logs/perf/$PARAM
  rm -rf $LOGS
  mkdir -p $LOGS
  (
    export LOGS
    export PARAM
    export PERF_STUDY="perf_$PARAM"
    rm -f ${PERF_STUDY}.e*
    qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d $HOME/deep-rl/ddpg perf_submit.sh
  )
done