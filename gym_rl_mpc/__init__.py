from gym.envs.registration import register
import numpy as np

DEFAULT_CONFIG = {
    "step_size": 0.1,
    "max_episode_time": 300,                    # Max time for episode [seconds]
    "crash_angle_condition": 10*(np.pi/180),
    "crash_omega_max": 30*(2*np.pi/60),
    "crash_omega_min": 1*(2*np.pi/60),
    "max_init_angle": 0,
    "reward_crash": -1e4,
    "max_wind_speed": 25,
    "min_wind_speed": 3,
    "gamma_theta": 0.04,                        # Exponential coefficient for platform_angle angle reward
    "gamma_omega": 0.11,                        # Exponential coefficient for omega reward
    "gamma_power": 0.1,                        # Exponential coefficient for power reward
    "reward_theta_dot": 0.4,                      # Coefficient for angular rate penalty
    "reward_control": 0,                        # Coefficient for control input penalty
    "reward_survival": 10,
}

register(
    id='PendulumStab-v1',
    entry_point='gym_rl_mpc.envs:PendulumEnv',
    kwargs={'env_config': DEFAULT_CONFIG}
)
