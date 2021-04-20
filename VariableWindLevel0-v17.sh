#!/bin/bash
#SBATCH -J RL_w_PSF                   # Sensible name for the job
#SBATCH -N 1                     # Allocate 2 nodes for the job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --gres=gpu:1
##SBATCH -c 32
#SBATCH -t 12:00:00          # Upper time limit for the job
#SBATCH -p GPUQ
module load fosscuda/2020b
module load Anaconda3/2020.07
conda init
conda env create -f environment.yml
conda activate gym-rl-mpc
python train.py --timesteps 3000000 --env VariableWindLevel0-v17 --psf --note "learning rate linear schedule, init_value=1e-4"lue=1e-4"