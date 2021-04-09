@echo off
python train.py --timesteps 5000000 --env VariableWindLevel0-v0 --note "omega crash max reduced. Only power reward. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel1-v0 --note "omega crash max reduced. Only power reward. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel2-v0 --note "omega crash max reduced. Only power reward. learning rate linear schedule, init_value=1e-4"