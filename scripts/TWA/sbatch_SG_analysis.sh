#!/bin/bash
#SBATCH --job-name=TWA_SG
#SBATCH --output=log_TWA_SG%a
#SBATCH --error=log_TWA_SG%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-24

# srun julia -t 48 TWA_SG2.jl ${SLURM_ARRAY_TASK_ID}
srun julia -t 48 TWA_SG_analysis.jl ${SLURM_ARRAY_TASK_ID}
