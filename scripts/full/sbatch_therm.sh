#!/bin/bash
#SBATCH --job-name=EDTherm
#SBATCH --output=log_EDTherm%a
#SBATCH --error=log_EDTherm%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-8

srun julia -t 48 Hfull_thermal.jl ${SLURM_ARRAY_TASK_ID}
