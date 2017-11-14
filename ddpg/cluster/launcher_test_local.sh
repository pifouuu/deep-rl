#!/bin/bash
for i in {1..3}
do
    /usr/bin/python3.5 /home/pierre/PycharmProjects/deep-rl/ddpg/ddpg.py --summary-dir /home/pierre/PycharmProjects/deep-rl/ddpg/results/ --max-episodes 20 --with-goal --episode-reset
done
