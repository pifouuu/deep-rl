#!/bin/bash
MAIN=/home/pierre/PycharmProjects/deep-rl/ddpg/ddpg.py
LOGDIR=/home/pierre/PycharmProjects/deep-rl/ddpg/results
for i in {1..1}
do
    /usr/bin/python3.5 $MAIN --summary-dir $LOGDIR --max-episodes 500 --episode-reset --with-goal --with-hindsight
done