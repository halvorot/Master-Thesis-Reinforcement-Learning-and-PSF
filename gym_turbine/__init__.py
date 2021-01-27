from gym.envs.registration import register
import numpy as np

DEFAULT_CONFIG = {
    "step_size": 0.01,
    "min_reward": -2000,
    "max_t_steps": 100000,
    "crash_angle_condition": 30*(np.pi/180),
    "max_init_angle": 15*(np.pi/180),
    "reward_crash": -1000,
    "n_actuators": 4,
    "n_observations": 2,                        # Pitch and roll angle
    "wind_dir": 0,
    "lambda_reward": 0.8,                       # Parameter for how much focus on stabilization vs power usage
    "gamma_p": 15,                              # Exponential coefficient for pitch angle reward
    "gamma_r": 15,                              # Exponential coefficient for roll angle reward
    "gamma_F": 1/4,                             # Coefficient for normalization of energy use reward
    "gamma_power": 1/4,                          # Coefficient for power use reward
}


register(
    id='TurbineStab-v0',
    entry_point='gym_turbine.envs:TurbineEnv',
    kwargs={'env_config': DEFAULT_CONFIG}
)
