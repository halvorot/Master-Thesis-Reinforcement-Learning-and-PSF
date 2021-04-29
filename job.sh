#!/bin/bash
srun --nodes=1 -c 32 --partition=GPUQ --time=24:00:00 --pty bash
module load fosscuda/2020b
conda activate gym-rl-mpc
python train.py --timesteps 10000000 --env VariableWindPSFtest-v17 --psf --note "learning rate linear schedule, init_value=1e-4" --num_cpus 32