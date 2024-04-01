#!/bin/bash
#SBATCH --job-name=TWA_SG
#SBATCH --output=log_TWA_SG%a
#SBATCH --error=log_TWA_SG%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-8

julia -t 48 TWA_SG.jl ${SLURM_ARRAY_TASK_ID}
