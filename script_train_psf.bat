@echo off

rem running on NTNU12382 26.04.2021
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 5 --note "Horizon results . psf_T 5"
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 10 --note "Horizon results . psf_T 10"
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 20 --note "Horizon results . psf_T 20"
python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 40 --note "Horizon results . psf_T 40"

rem running on NTNU16535 26.04.2021
rem python train.py --timesteps 2000000 --env VariableWindLevel10-v17 --psf --psf_T 10 --note "Horizon results . psf_T 10"
rem python train.py --timesteps 2000000 --env VariableWindLevel10-v17 --psf --psf_T 20 --note "Horizon results . psf_T 20"


