#!/bin/bash
#SBATCH --job-name=TWA_SG_run
#SBATCH --output=log_TWA_SG_run%a
#SBATCH --error=log_TWA_SG_run%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-12

srun julia -t 48 TWA_SG3.jl ${SLURM_ARRAY_TASK_ID}
