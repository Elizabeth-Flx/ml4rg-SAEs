#!/bin/bash
#
# SLURM resource specifications
#
#
#SBATCH --job-name=sae_hparam_grid_search_project2_group1
#SBATCH --time=4-23:59:59
#SBATCH --nodes=1         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4      
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err












