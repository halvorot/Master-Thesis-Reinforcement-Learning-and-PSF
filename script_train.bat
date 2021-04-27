@echo off
python train.py --timesteps 5000000 --env VariableWindLevel0-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel1-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel2-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel3-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel4-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel5-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel6-v17   --note "baseline "
python train.py --timesteps 5000000 --env VariableWindLevel7-v17   --note "baseline "