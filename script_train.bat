@echo off
python train.py --timesteps 10000000 --env VariableWindLevel0-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel1-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel2-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel3-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel4-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel5-v17 --note "learning rate linear schedule, init_value=1e-4"