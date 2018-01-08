#!/usr/bin/env bash

MEMORIES=(hsarst)
STRATS=(future final)
SAMPLERS=(intervalC comp)
ALPHAS=(0.1 1 2 5)
DELTAS=(inf)
ACTIVATIONS=(tanh)
IVGS=(True)
TCLIPS=(True)
NSTEPS=(200)
EXPLOS=(0 0.15 0.3)

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
                                    for EXPLO in ${EXPLOS[*]}
                                    do
                                        export LOGS=logs/perf/${MEMORY}_${STRAT}_${SAMPLER}_${ALPHA}_${DELTA}_${ACTIVATION}_${IVG}_${TCLIP}_${NSTEP}_${EXPLO}
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
                                            export EXPLO
                                            export PERF_STUDY="perf_${MEMORY}_${STRAT}_${SAMPLER}_${ALPHA}_${DELTA}_${ACTIVATION}_${IVG}_${TCLIP}_${NSTEP}_${EXPLO}"
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
done
