#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=32768M   # memory per CPU core
#SBATCH -J "test_resources_gender_training"   # job name
#SBATCH --qos=test


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/grphome/fslg_census/compute/envs/torch/bin/python /grphome/fslg_census/compute/machine_learning_models/classification_models/branches/jackson/RLL_classifiers_pytorch/src/train_example.py