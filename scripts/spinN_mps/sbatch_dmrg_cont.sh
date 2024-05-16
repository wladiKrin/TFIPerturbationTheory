#!/bin/bash
#SBATCH   --job-name=dmrg8128_cont
#SBATCH --output=log_dmrg8128_cont%a
#SBATCH  --error=log_dmrg8128_cont%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=1-5

julia -t 48 dmrg.jl 8 128 ${SLURM_ARRAY_TASK_ID}
