#!/bin/bash
#SBATCH --job-name=fullSpec
#SBATCH --output=log_fullSpec%a
#SBATCH --error=log_fullSpec%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-12

srun julia -t 48 HfullRed_spec.jl ${SLURM_ARRAY_TASK_ID}
