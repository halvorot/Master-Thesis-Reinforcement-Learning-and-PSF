@echo off
python train.py --timesteps 5000000 --env VariableWindLevel1-v17 --psf --psf_T 20 --psf_lb_omega 0.41887902048 --psf_ub_omega 0.94247779608 --note "learning rate linear schedule, init_value=1e-4,--T 20 --psf-bounds 4-9 --note"
python train.py --timesteps 5000000 --env VariableWindLevel1-v17 --psf --psf_T 40 --note "learning rate linear schedule, init_value=1e-4, --T 40 --psf-bounds 5-7.6"
python train.py --timesteps 5000000 --env VariableWindLevel1-v17 --psf --psf_T 5 --note "learning rate linear schedule, init_value=1e-4, --T 5 --psf-bounds 5-7.6"