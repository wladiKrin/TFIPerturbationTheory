#!/bin/bash
#SBATCH --job-name=HardcBosSpec4
#SBATCH --output=log_HardcBosSpec4%a
#SBATCH --error=log_HardcBosSpec4%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-1

srun julia -t 48 Heff_hardcBosonsSpec.jl ${SLURM_ARRAY_TASK_ID}
