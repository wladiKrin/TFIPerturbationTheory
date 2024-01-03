#!/bin/bash
#SBATCH --job-name=effPert
#SBATCH --output=log_effPert%a
#SBATCH --error=log_effPert%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-6

srun julia -t 48 Heff_perturbative.jl ${SLURM_ARRAY_TASK_ID}
