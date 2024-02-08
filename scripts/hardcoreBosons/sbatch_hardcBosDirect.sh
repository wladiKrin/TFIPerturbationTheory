#!/bin/bash
#SBATCH --job-name=HardcBosD87
#SBATCH --output=log_HardcBosD87%a
#SBATCH --error=log_HardcBosD87%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=5-5

srun julia -t 48 Heff_hardcBosonsDirectTime.jl ${SLURM_ARRAY_TASK_ID}
