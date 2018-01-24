#!/usr/bin/env bash

ENVTS=(CMCPos-v0)
MEMORIES=(sarst)
STRATS=(future)
SAMPLERS=(rnd)
ALPHAS=(2)
DELTAS=(inf)
ACTIVATIONS=(linear)
IVGS=(True)
TCLIPS=(True)
SIGMAS=(2)

for ENVT in ${ENVTS[*]}
do
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
                                    for SIGMA in ${SIGMAS[*]}
                                    do
                                        export LOGS=./log/logs/${ENVT}_${MEMORY}_${STRAT}_${SAMPLER}_${ALPHA}_${DELTA}_${ACTIVATION}_${IVG}_${TCLIP}_${SIGMA}
                                        rm -rf $LOGS
                                        mkdir -p $LOGS
                                        (
                                            export LOGS
                                            export ENVT
                                            export MEMORY
                                            export STRAT
                                            export SAMPLER
                                            export ALPHA
                                            export DELTA
                                            export ACTIVATION
                                            export IVG
                                            export TCLIP
                                            export SIGMA
                                            export PERF_STUDY="perf_${ENVT}_${MEMORY}_${STRAT}_${SAMPLER}_${ALPHA}_${DELTA}_${ACTIVATION}_${IVG}_${TCLIP}_${SIGMA}"
                                            rm -f ${PERF_STUDY}.e*
                                            qsub -N ${PERF_STUDY} -o "$LOGS/${PERF_STUDY}.out" -b "$LOGS/${PERF_STUDY}.err" -d $HOME/deep-rl perf_submit.sh
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
