@echo off

rem running on NTNU12382 26.04.2021
rem python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 5 --note "Horizon results . psf_T 5"
rem python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 10 --note "Horizon results . psf_T 10"
rem python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 20 --note "Horizon results . psf_T 20"
rem python train.py --timesteps 2000000 --env VariableWindLevel1-v17 --psf --psf_T 40 --note "Horizon results . psf_T 40"

rem running on NTNU16535 26.04.2021
rem python train.py --timesteps 2000000 --env VariableWindLevel10-v17 --psf --psf_T 10 --note "Horizon results . psf_T 10"
rem python train.py --timesteps 2000000 --env VariableWindLevel10-v17 --psf --psf_T 20 --note "Horizon results . psf_T 20"

rem running somewhere
rem python train.py --timesteps 2000000 --env VariableWindLevel11-v17 --psf --note "error zone"
rem python train.py --timesteps 2000000 --env VariableWindLevel12-v17 --psf --note "error zone"

python train.py --timesteps 2000000 --env VariableWindLevel13-v17 --note --psf "error zone"
python train.py --timesteps 2000000 --env VariableWindLevel14-v17 --note --psf "error zone"
