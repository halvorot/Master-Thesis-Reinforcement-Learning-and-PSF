#!/bin/bash
#SBATCH -J job                   # Sensible name for the job
#SBATCH -N 1                     # Allocate 2 nodes for the job
#SBATCH --ntasks-per-node=1     # 1 task per node
#SBATCH --gres=gpu:1
##SBATCH -c 64
#SBATCH -t 12:00:00          # Upper time limit for the job
#SBATCH -p GPUQ
module load fosscuda/2020b
module load Anaconda3/2020.07
conda env -f environment.yml
conda activate gym-rl-mpc
python train.py --timesteps 3000000 --env VariableWindLevel0-v16 --psf --note "learning rate linear schedule, init_value=1e-4"


