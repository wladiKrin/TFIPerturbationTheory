#!/bin/bash
#SBATCH --job-name=HardcBosD
#SBATCH --output=log_HardcBosD%a
#SBATCH --error=log_HardcBosD%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-7

srun julia -t 48 Heff_hardcBosonsDirect.jl ${SLURM_ARRAY_TASK_ID}
