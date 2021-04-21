@echo off
python train.py --timesteps 6000000 --env VariableWindLevel0-v17 --psf --note "tighter constraints. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 6000000 --env VariableWindLevel1-v17 --psf --note "tighter constraints. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 6000000 --env VariableWindLevel2-v17 --psf --note "tighter constraints. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 6000000 --env VariableWindLevel3-v17 --psf --note "tighter constraints. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 6000000 --env VariableWindLevel4-v17 --psf --note "tighter constraints. learning rate linear schedule, init_value=1e-4"