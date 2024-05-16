#!/bin/bash
#SBATCH   --job-name=tdvpMin832
#SBATCH --output=log_tdvpMin832%a
#SBATCH  --error=log_tdvpMin832%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=1-1

julia -t 48 tdvp2.jl 8 32 ${SLURM_ARRAY_TASK_ID}
