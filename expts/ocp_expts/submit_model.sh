#!/bin/bash

## job name
#SBATCH --job-name=test
#SBATCH --output=stdout.out
##SBATCH --error=stderr.err

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=6g
#SBATCH --time=16:00:00

/private/home/abhshkdz/.conda/envs/ocp/bin/python /private/home/mshuaibi/baselines/scripts/ocp_bench.py
