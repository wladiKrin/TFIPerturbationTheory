#!/bin/bash
#SBATCH --job-name=HardcBosSpec
#SBATCH --output=log_HardcBosSpec%a
#SBATCH --error=log_HardcBosSpec%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-3

srun julia -t 48 Heff_hardcBosonsSpec.jl ${SLURM_ARRAY_TASK_ID}
