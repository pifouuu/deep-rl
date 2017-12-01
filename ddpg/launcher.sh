#!/bin/bash

# Example submission of PBS jobs in a loop

NUMBERS=$(seq 1 20) # Create an array of seed values from 1 to NSEEDS
DDPG=/home/fournier/deep-rl/ddpg/main.py
LOGDIR=/home/fournier/deep-rl/ddpg/results/


# Loop through the different seed values and submit a run for each

for NUM in ${NUMBERS}
do
	# set the job name
	NAME=hindsightRandomGoal_${NUM}
	echo "Submitting: ${NAME}"

	# Build a string called PBS which contains the instructions for your run
	# This requests 1 node for 1 hour. Runs a program called "my_program" with an argument.

	PBS="#!/bin/bash\n\
	#PBS -N ${NAME}\n\
	#PBS -l nodes=1:ppn=1\n\
	#PBS -l walltime=12:00:00\n\
	#PBS -o out/${NAME}.out\n\
	#PBS -e err/${NAME}.err\n\
	cd \$PBS_O_WORKDIR\n\
	/usr/bin/python3.4 $DDPG --summary-dir $LOGDIR --wrapper WithGoal --memory hindsight_ep --sampler Random"

	# Note that $PBS_O_WORKDIR is escaped ("\"). We don't want bash to evaluate this variable right now. Instead it will be evaluated when the command runs on the node.

	# Echo the string PBS to the function qsub, which submits it as a cluster job for you
	# A small delay is included to avoid overloading the submission process

	echo -e ${PBS} | qsub
	sleep 2
	echo "done."

done
