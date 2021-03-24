from gym.envs.registration import register
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM, RPM2RAD, DEG2RAD

DEFAULT_CONFIG = {
    "use_psf": False,
    "step_size": 0.1,
    "max_episode_time": 300,                    # Max time for episode [seconds]
    "crash_angle_condition": 10*DEG2RAD,
    "crash_omega_max": 15*RPM2RAD,
    "crash_omega_min": 3*RPM2RAD,
    "max_wind_speed": 25,
    "min_wind_speed": 5,
    "gamma_theta": 0.12,                        # Exponential coefficient for platform_angle angle reward
    "gamma_omega": 0.285,                        # Exponential coefficient for omega reward
    "gamma_power": 0.1,                        # Exponential coefficient for power reward
    "gamma_theta_dot": 3,                      # Coefficient for angular rate penalty
    "gamma_psf": 5,
    "gamma_input": 0,                        # Coefficient for control input penalty
    "reward_survival": 1,
}

VARIABLE_WIND_CONFIG = DEFAULT_CONFIG.copy()
VARIABLE_WIND_CONFIG["wind_period"] = 60
VARIABLE_WIND_CONFIG["max_wind_amplitude"] = 3
VARIABLE_WIND_CONFIG["wind_noise"] = False

SCENARIOS = {
    'ConstantWind-v1': {   
        'entry_point': 'gym_rl_mpc.envs:ConstantWind',
        'config': DEFAULT_CONFIG
    },
    'VariableWind-v1': {
        'entry_point': 'gym_rl_mpc.envs:VariableWind',
        'config': VARIABLE_WIND_CONFIG
    }
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        kwargs={'env_config': SCENARIOS[scenario]['config']}
    )