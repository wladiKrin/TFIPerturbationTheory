#!/bin/bash
#SBATCH --job-name=HardcBos5
#SBATCH --output=log_HardcBos5%a
#SBATCH --error=log_HardcBos5%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-7

srun julia -t 48 Heff_hardcBosons.jl ${SLURM_ARRAY_TASK_ID}
