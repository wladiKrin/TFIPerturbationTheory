#!/bin/bash
#SBATCH   --job-name=tdvp3232
#SBATCH --output=log_tdvp3232%a
#SBATCH  --error=log_tdvp3232%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=4-8

julia -t 48 tdvp2.jl 32 32 ${SLURM_ARRAY_TASK_ID}
