import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.append(HERE.parent) # to import gym and psf

from PSF.PSF import PSF
import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params

number_of_iter = 50
np.random.seed(42)
psf = PSF(sys={"xdot": sym.symbolic_x_dot,
               "x": sym.x,
               "u": sym.u,
               "p": sym.w,
               "Hx": sym.Hx,
               "hx": sym.hx,
               "Hu": sym.Hu,
               "hu": sym.hu
               }, N=20, T=20, R=np.diag([
    1 / params.max_thrust_force ** 2,
    1 / params.max_blade_pitch ** 2,
    1 / params.max_power_generation ** 2
]), PK_path=Path("../PSF", "stored_PK"), lin_bounds={"w": [3,
                                                           25],
                                                     "u_p": [0 * params.max_blade_pitch, params.max_blade_pitch],
                                                     "Omega": [params.omega_setpoint(3),
                                                               params.omega_setpoint(25)],
                                                     "P_ref": [0, params.max_power_generation],
                                                     "theta": [-10 * sym.DEG2RAD, 10 * sym.DEG2RAD],
                                                     "theta_dot": [-45 * sym.DEG2RAD, 45 * sym.DEG2RAD]
                                                     }, slew_rate=[1e5, 8 * params.DEG2RAD, 1e6], LP_flag=False,
          slack_flag=True, terminal_flag=True, jit_flag=False)
start = time.time()
for i in range(number_of_iter):
    print(psf.calc(x=[np.random.uniform(low=-7, high=9) * params.DEG2RAD,
                      0,
                      np.random.uniform(low=5.1, high=7.5) * params.RPM2RAD],
                   u_L=[
                       2 * np.random.uniform(low=-params.max_thrust_force, high=params.max_thrust_force),
                       2 * np.random.uniform(low=-4 * params.DEG2RAD, high=params.max_blade_pitch),
                       2 * np.random.uniform(low=0, high=params.max_power_generation)
                   ],
                   u_stable=[0, 0, 1e6],
                   u_prev=[0, 0, 0],
                   ext_params=np.random.uniform(low=3, high=25),
                   reset_x0=True
                   ))

end = time.time()
print(
    f"Jit: {psf.jit_flag}. Solved {number_of_iter} iterations in {end - start} s, [{(end - start) / number_of_iter} s/step]")
