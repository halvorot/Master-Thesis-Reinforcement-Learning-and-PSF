@echo off
python train.py --timesteps 5000000 --env VariableWindLevel0-v16 --note "prev_wind_speed added to obs. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel1-v16 --note "prev_wind_speed added to obs. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel2-v16 --note "prev_wind_speed added to obs. learning rate linear schedule, init_value=1e-4"
python train.py --timesteps 5000000 --env VariableWindLevel3-v16 --note "prev_wind_speed added to obs. learning rate linear schedule, init_value=1e-4"