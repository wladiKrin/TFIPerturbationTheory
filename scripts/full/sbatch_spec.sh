#!/bin/bash
#SBATCH --job-name=fullBSpec
#SBATCH --output=log_fullBSpec%a
#SBATCH --error=log_fullBSpec%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-17

srun julia -t 48 HfullBound_spec.jl ${SLURM_ARRAY_TASK_ID}
