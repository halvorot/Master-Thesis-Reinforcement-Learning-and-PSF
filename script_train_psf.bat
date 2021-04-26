@echo off
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 5 --note "Horizon results . psf_T 5"
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 10 --note "Horizon results . psf_T 10"
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 20 --note "Horizon results . psf_T 20"
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 40 --note "Horizon results . psf_T 40"
