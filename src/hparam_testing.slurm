#!/bin/bash
#
# SLURM resource specifications
#
#
#SBATCH --job-name=sae_weep_p2_g1
#SBATCH --time=02:00:00
#SBATCH --nodes=1         
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --partition=student_project


wandb agent annisch-technical-university-of-munich/batchtopk_comparison/utpk1t8l












