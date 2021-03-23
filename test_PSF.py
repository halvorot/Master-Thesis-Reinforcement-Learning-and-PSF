import time
from pathlib import Path

import numpy as np
from casadi import vertcat

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
               },
          N=200,
          T=10,
          R=np.diag([
              1 / params.max_thrust_force ** 2,
              1 / params.max_blade_pitch ** 2,
              1 / params.max_power_generation ** 2
          ]),
          PK_path=Path("PSF", "stored_PK"),
          lin_bounds={"w": [3,
                            25],
                      "u_p": [0 * params.max_blade_pitch, params.max_blade_pitch],
                      "Omega": [params.omega_setpoint(3),
                                params.omega_setpoint(25)],
                      "P_ref": [0, params.max_power_generation],
                      "theta": [-10 * sym.DEG2RAD, 10 * sym.DEG2RAD],
                      "theta_dot": [-45 * sym.DEG2RAD, 45 * sym.DEG2RAD]
                      },
          LP_flag=True,
          slack_flag=True,
          jit_flag=False,
          terminal_flag=False
          )
start = time.time()
for i in range(number_of_iter):
    print(psf.calc(x=[np.random.uniform(low=-7, high=10) * params.DEG2RAD,
                      0,
                      np.random.uniform(low=5, high=7.5) * params.RPM2RAD],
                   u_L=[
                       2 * np.random.uniform(low=-params.max_thrust_force, high=params.max_thrust_force),
                       2 * np.random.uniform(low=-4 * params.DEG2RAD, high=params.max_blade_pitch),
                       2 * np.random.uniform(low=0, high=params.max_power_generation)
                   ],
                   u0=vertcat([500000, 0, 14e6]),
                   ext_params=vertcat(np.random.uniform(low=15, high=20)),
                   reset_x0=True

                   ))

end = time.time()
print(
    f"Jit: {psf.jit_flag}. Solved {number_of_iter} iterations in {end - start} s, [{(end - start) / number_of_iter} s/step]")
