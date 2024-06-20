#!/bin/bash
#SBATCH --job-name=TWA_SG_runFockVar
#SBATCH --output=log_TWA_SG_runFockVar%a
#SBATCH --error=log_TWA_SG_runFockVar%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-4

srun julia -t 48 TWA_SG4.jl ${SLURM_ARRAY_TASK_ID}
