#!/bin/bash
#SBATCH --job-name=GS
#SBATCH --output=log_GS%a
#SBATCH --error=log_GS%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-7

srun julia -t 48 tebd_gs_search.jl 8 ${SLURM_ARRAY_TASK_ID}
