#!/bin/bash
#SBATCH   --job-name=tdvpL50B64S4
#SBATCH --output=log_tdvpL50B64S4g%a
#SBATCH  --error=log_tdvpL50B64S4g%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=1-1

julia -t 48 tdvp2.jl 50 64 ${SLURM_ARRAY_TASK_ID} 4
