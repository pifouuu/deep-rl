#!/bin/bash
MAIN=/home/pierre/PycharmProjects/deep-rl/ddpg/main.py
LOGDIR=/home/pierre/PycharmProjects/deep-rl/ddpg/results
for i in {1..1}
do
    /usr/bin/python3.5 $MAIN --summary-dir $LOGDIR --wrapper NoGoal --memory SAS --sampler NoGoal
done