#!/usr/bin/env bash

MEMORIES=(sarst)
STRATS=(final)
SAMPLERS=(intervalC goalC)
ALPHAS=(0.1 1 5)
DELTAS=(inf)
ACTIVATIONS=(tanh linear)
IVGS=(False True)
TCLIPS=(False True)
NSTEPS=(200)

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
                                for NSTEP in ${NSTEPS[*]}
                                do
                                    export LOGS=logs/perf/${MEMORY}_${STRAT}_${SAMPLER}_${ALPHA}_${DELTA}_${ACTIVATION}_${IVG}_${TCLIP}_${NSTEP}
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
                                        export NSTEP
                                        export PERF_STUDY="perf_${MEMORY}_${STRAT}_${SAMPLER}_${ALPHA}_${DELTA}_${ACTIVATION}_${IVG}_${TCLIP}_${NSTEP}"
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
done
