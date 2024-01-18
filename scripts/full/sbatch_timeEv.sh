#!/bin/bash
#SBATCH --job-name=EDTimeEvol
#SBATCH --output=log_EDTimeEvol%a
#SBATCH --error=log_EDTimeEvol%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-16

srun julia -t 48 Hfull_timeEvol.jl ${SLURM_ARRAY_TASK_ID}
