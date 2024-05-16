#!/bin/bash
#SBATCH   --job-name=dmrg248256
#SBATCH --output=log_dmrg248256%a
#SBATCH  --error=log_dmrg248256%a
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24
#SBATCH --array=1-12

julia -t 48 dmrg.jl 24 8 256 ${SLURM_ARRAY_TASK_ID}
