from gym.envs.registration import register
import numpy as np
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM, RPM2RAD, DEG2RAD

DEFAULT_CONFIG = {
    "use_psf": False,
    "step_size": 0.1,
    "max_episode_time": 300,                    # Max time for episode [seconds]
    "working_range_reward_multiplier": 10,
    "crash_angle_condition": 10*DEG2RAD,
    "crash_omega_max": 15*RPM2RAD,
    "crash_omega_min": 3*RPM2RAD,
    "max_wind_speed": 20,
    "min_wind_speed": 10,
    "gamma_theta": 0.12,                        # Exponential coefficient for platform_angle angle reward
    "gamma_omega": 0.285,                        # Exponential coefficient for omega reward
    "gamma_power": 0.1,                        # Exponential coefficient for power reward
    "reward_theta_dot": 3,                      # Coefficient for angular rate penalty
    "reward_control": 0,                        # Coefficient for control input penalty
    "reward_survival": 1,
}

register(
    id='TurbineStab-v0',
    entry_point='gym_rl_mpc.envs:TurbineEnv',
    kwargs={'env_config': DEFAULT_CONFIG}
)
