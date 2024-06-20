#!/bin/bash
#SBATCH   --job-name=timesteps
#SBATCH --output=log_timesteps
#SBATCH  --error=log_timesteps
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
##SBATCH --cores-per-socket=24

julia -t 48 tdvp_timestep_analysis.jl
