#!/bin/bash
#SBATCH   --job-name=dmrg1004
#SBATCH --output=log_dmrg1004%a
#SBATCH  --error=log_dmrg1004%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=1-11

# julia -t 48 dmrg2.jl 100 ${SLURM_ARRAY_TASK_ID} 1
julia -t 48 dmrg2.jl 100 4 ${SLURM_ARRAY_TASK_ID}
