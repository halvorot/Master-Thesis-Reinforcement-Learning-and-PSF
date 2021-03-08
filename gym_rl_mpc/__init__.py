from gym.envs.registration import register
import numpy as np

DEFAULT_CONFIG = {
    "step_size": 0.5,
    "min_reward": -1500,
    "max_t_steps": 3000,
    "crash_angle_condition": 100*(np.pi/180),
    "crash_omega_condition": 300*(np.pi/180),
    "max_init_angle": 0*3*(np.pi/180),
    "reward_crash": -2000,
    "max_wind_speed": 10,
    "gamma": 50,                                # Exponential coefficient for platform_angle angle reward
    "reward_theta_dot": 20,                      # Coefficient for angular rate penalty
    "reward_omega": 1,                          # Coefficient for control input penalty
    "reward_control": 0.2,                        # Coefficient for control input penalty
}

register(
    id='PendulumStab-v1',
    entry_point='gym_rl_mpc.envs:PendulumEnv',
    kwargs={'env_config': DEFAULT_CONFIG}
)
