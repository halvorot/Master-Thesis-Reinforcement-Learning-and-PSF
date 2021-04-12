@echo off
python train.py --timesteps 5000000 --env VariableWindLevel4-v16 --note "with omega_dot in obs. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel6-v16 --note "with omega_dot in obs. learning rate linear schedule, init_value=1e-4"