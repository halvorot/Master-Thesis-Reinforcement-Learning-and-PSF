import time

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
          N=20,
          T=20,
          R=np.diag([
              1 / params.max_thrust_force ** 2,
              1 / params.max_blade_pitch ** 2,
              1 / params.max_power_generation ** 2
          ]),
          slack=True,
          jit_flag=False
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
                   u0=vertcat([500000, 0, 10e6]),
                   ext_params=vertcat(np.random.uniform(low=15, high=15)),
                   reset_x0=True
                   ))

end = time.time()
print(
    f"Jit: {psf.jit_flag}. Solved {number_of_iter} iterations in {end - start} s, [{(end - start) / number_of_iter} s/step]")


