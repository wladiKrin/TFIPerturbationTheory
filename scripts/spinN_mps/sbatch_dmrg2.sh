#!/bin/bash
#SBATCH   --job-name=dmrg1006
#SBATCH --output=log_dmrg1006%a
#SBATCH  --error=log_dmrg1006%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=1-7

# julia -t 48 dmrg2.jl 100 ${SLURM_ARRAY_TASK_ID} 1
julia -t 48 dmrg2.jl 100 6 ${SLURM_ARRAY_TASK_ID}
