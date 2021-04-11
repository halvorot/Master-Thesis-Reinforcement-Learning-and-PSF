@echo off
python train.py --timesteps 5000000 --env VariableWindLevel0-v10 --note "omega_dot added to obs. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel1-v10 --note "omega_dot added to obs. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel2-v10 --note "omega_dot added to obs. learning rate linear schedule, init_value=1e-4"