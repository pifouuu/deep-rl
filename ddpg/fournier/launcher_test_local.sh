#!/bin/bash
MAIN=/home/pierre/PycharmProjects/deep-rl/ddpg/ddpg.py
LOGDIR=/home/pierre/PycharmProjects/deep-rl/ddpg/results
for i in {1..20}
do
    /usr/bin/python3.5 $MAIN --delta 1 --summary-dir $LOGDIR --max-episodes 500 --episode-reset --with-goal --with-hindsight
done