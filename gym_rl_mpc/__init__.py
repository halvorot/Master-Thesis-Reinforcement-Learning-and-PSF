from gym.envs.registration import register
import numpy as np

DEFAULT_CONFIG = {
    "step_size": 1,
    "min_reward": -1500,
    "max_t_steps": 3000,
    "crash_angle_condition": 7*(np.pi/180),
    "max_init_angle": 3*(np.pi/180),
    "reward_crash": -2000,
    "gamma": 50,                                # Exponential coefficient for angle angle reward
    "reward_theta_dot": 1,                      # Coefficient for angular rate penalty
    "reward_control": 1,                      # Coefficient for control input penalty
}

register(
    id='PendulumStab-v0',
    entry_point='gym_rl_mpc.envs:PendulumEnv',
    kwargs={'env_config': DEFAULT_CONFIG}
)
