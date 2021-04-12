@echo off
python train.py --timesteps 3000000 --env VariableWindLevel0-v26 --psf
python train.py --timesteps 3000000 --env VariableWindLevel1-v26 --psf
python train.py --timesteps 3000000 --env VariableWindLevel2-v26 --psf
python train.py --timesteps 3000000 --env VariableWindLevel3-v26 --psf