#!/bin/bash
#SBATCH --job-name=effSectors
#SBATCH --output=log_effSectors%a
#SBATCH --error=log_effSectors%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-4

srun julia -t 48 Heff_sectors.jl ${SLURM_ARRAY_TASK_ID}
