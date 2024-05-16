#!/bin/bash
#SBATCH   --job-name=tdvp16128
#SBATCH --output=log_tdvp16128%a
#SBATCH  --error=log_tdvp16128%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=2-3

julia -t 48 tdvp2.jl 16 128 ${SLURM_ARRAY_TASK_ID}
