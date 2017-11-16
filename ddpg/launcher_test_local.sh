#!/bin/bash
MAIN=/home/pierre/PycharmProjects/deep-rl/ddpg/ddpg.py
LOGDIR=/home/pierre/PycharmProjects/deep-rl/ddpg/results/
for i in {1..20}
do
    /usr/bin/python3.5 $MAIN --summary-dir $LOGDIR --delta 1 --max-episodes 500 --episode-reset
done