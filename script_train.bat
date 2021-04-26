@echo off
python train.py --timesteps 10000000 --env VariableWindLevel0-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel0-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel0-v17 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel0-v13 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel0-v14 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel0-v15 --note "learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 10000000 --env VariableWindLevel0-v16 --note "learning rate linear schedule, init_value=1e-4"
