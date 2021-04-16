import os
import sys
from pathlib import Path
import logging
import time
import numpy as np

HERE = Path(__file__).parent
sys.path.append(HERE.parent)  # to import gym and psf
os.chdir(HERE.parent)
from PSF.PSF import PSF
import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params

logging.basicConfig(level=logging.DEBUG)

print("Test Started")
number_of_state_perm = 3
number_of_input_perm = 20
np.random.seed(42)
kwargs = dict(sys=sym.get_sys(),
              N=200,
              T=100,
              t_sys=sym.get_terminal_sys(),
              R=np.diag([
                  1 / params.max_thrust_force ** 2,
                  1 / params.max_blade_pitch ** 2,
                  1 / params.max_power_generation ** 2
              ]),
              PK_path=Path(HERE, "terminalset"),
              ext_step_size=0.1,
              slew_rate=[1e5, 8 * params.DEG2RAD, 1e6],
              slack_flag=True)

psf_HF = PSF(**kwargs)


psf_LF = PSF(**kwargs)

nx = sym.x.shape[0]
nu = sym.u.shape[0]
start = time.time()
for j in range(number_of_state_perm):
    logging.debug(f"State permute: {j}")
    x = [np.random.uniform(low=0, high=7) * params.DEG2RAD,
         0,
         np.random.uniform(low=5.1, high=7.5) * params.RPM2RAD]

    w=np.random.uniform(low=5, high=25 * 2 / 3)
    for i in range(number_of_input_perm):
        logging.debug(f"Input permute: {i}")
        u_L = [
            np.random.uniform(low=-params.max_thrust_force, high=params.max_thrust_force),
            np.random.uniform(low=-4 * params.DEG2RAD, high=params.max_blade_pitch),
            np.random.uniform(low=0, high=params.max_power_generation)
        ]
        kwargs_calc = dict(x=x,
                           u_L=u_L,
                           u_prev=u_L,
                           ext_params=w,
                           reset_x0=True
                           )
        logging.info(kwargs_calc)
        logging.info(psf_LF.calc(**kwargs_calc))

end = time.time()
number_of_iter = number_of_input_perm * number_of_state_perm
print(f"Number of state permutations: {number_of_state_perm}\n Number of input permutations: {number_of_input_perm}")
print(f"Solved {number_of_iter} iterations in {end - start} s. {(end - start) / number_of_iter} s/step]")
