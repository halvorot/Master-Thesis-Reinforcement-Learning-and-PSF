import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PSF.PSF import PSF
from gym_rl_mpc import DEFAULT_CONFIG

import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params

PI = 3.14
df = pd.read_csv(Path("logs", "debug", "crash_data.csv"))

states = ["theta", "theta_dot","omega", "wind_speed", "adjusted_wind_speed"]

rad_to_deg_list = ["theta", "theta_dot"]
rad_to_rpm_list = ["omega"]
df[rad_to_deg_list] = df[rad_to_deg_list] * 360 / (2 * PI)
df[rad_to_rpm_list] = df[rad_to_rpm_list] * 60 / (2 * PI)

df[states].plot()


df.drop(labels=states, axis=1).plot()
plt.show()

sys = {
    "xdot": sym.symbolic_x_dot,
    "x": sym.x,
    "u": sym.u,
    "p": sym.w,
    "Hx": sym.Hx,
    "hx": sym.hx,
    "Hu": sym.Hu,
    "hu": sym.hu
}
R = np.diag(
    [
        1 / params.max_thrust_force ** 2,
        1 / params.max_blade_pitch ** 2,
        1 / params.max_power_generation ** 2
    ])
actuation_max_rate = [params.max_thrust_rate, params.max_blade_pitch_rate, params.max_power_rate]

## PSF init ##
psf = PSF(sys=sys, N=20, T=2, R=R, PK_path=Path("PSF", "stored_PK"), slew_rate=actuation_max_rate,
          ext_step_size=DEFAULT_CONFIG["step_size"], slack_flag=True)

F_thr = action[0] * params.max_thrust_force
blade_pitch = action[1] * params.max_blade_pitch
power = action[2] * params.max_power_generation
action_un_normalized = [F_thr, blade_pitch, power]
new_adjusted_wind_speed = params.wind_inflow_ratio * self.wind_speed - params.L * np.cos(
    self.turbine.platform_angle) * self.turbine.state[1]
psf_params = [new_adjusted_wind_speed]
u_prev = self.turbine.input