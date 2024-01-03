#!/bin/bash
#SBATCH --job-name=effDirect
#SBATCH --output=log_effDirect%a
#SBATCH --error=log_effDirect%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-6

srun julia -t 48 Heff_directStates.jl ${SLURM_ARRAY_TASK_ID}
