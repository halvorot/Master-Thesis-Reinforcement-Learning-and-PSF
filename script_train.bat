@echo off
python train.py --timesteps 5000000 --env VariableWindLevel0-v26 --note "omega_dot added to obs. omega_dot reward added. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel1-v26 --note "omega_dot added to obs. omega_dot reward added. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel2-v26 --note "omega_dot added to obs. omega_dot reward added. learning rate linear schedule, init_value=1e-4"