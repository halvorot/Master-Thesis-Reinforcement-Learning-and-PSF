@echo off
python train.py --timesteps 20000000 --env VariableWindLevel0-v0
python train.py --timesteps 20000000 --env VariableWindLevel1-v0
python train.py --timesteps 20000000 --env VariableWindLevel2-v0
python train.py --timesteps 20000000 --env VariableWindLevel3-v0