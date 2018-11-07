#!/bin/bash
# This script runs multiple batch jobs to test different hyperparameter
# settings. For each setting, it creates a different PBS file and calls
# it.

# MODIFY THESE
declare training_file="gan2.py"
declare walltime="48:00:00"
declare jobname="DCNN_training"
declare netid="yuyunli2"
declare directory="~/scratch/deep-learning/hw7/1b/"

# Declare the hyperparameters you want to iterate over
declare -a trial_number=(0)

# For each parameter setting we generate a new PBS file and run it
for trial in "${trial_number[@]}"
do
  python BOW_sentiment_analysis.py $training_file $walltime $jobname $netid $directory $trial > run.pbs
  echo "Submitting $trial"
  qsub run.pbs -A bauh
done
