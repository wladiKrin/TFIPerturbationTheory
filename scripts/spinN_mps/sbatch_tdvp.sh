#!/bin/bash
#SBATCH   --job-name=tdvp2091red
#SBATCH --output=log_tdvp2091red%a
#SBATCH  --error=log_tdvp2091red%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --array=1-8

julia -t 48 tdvp.jl 20 91 ${SLURM_ARRAY_TASK_ID}
