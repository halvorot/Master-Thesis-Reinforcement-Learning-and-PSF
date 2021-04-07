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

VARIABLE_WIND_CONFIG_lvl1 = VARIABLE_WIND_CONFIG.copy()
VARIABLE_WIND_CONFIG_lvl1["max_wind_amplitude"] = 1
VARIABLE_WIND_CONFIG_lvl1["max_wind_speed"] = 20
VARIABLE_WIND_CONFIG_lvl1["min_wind_speed"] = 10

VARIABLE_WIND_CONFIG_lvl2 = VARIABLE_WIND_CONFIG.copy()
VARIABLE_WIND_CONFIG_lvl2["max_wind_amplitude"] = 1
VARIABLE_WIND_CONFIG_lvl2["max_wind_speed"] = 25
VARIABLE_WIND_CONFIG_lvl2["min_wind_speed"] = 5

VARIABLE_WIND_CONFIG_lvl3 = VARIABLE_WIND_CONFIG.copy()
VARIABLE_WIND_CONFIG_lvl3["max_wind_amplitude"] = 3
VARIABLE_WIND_CONFIG_lvl3["max_wind_speed"] = 25
VARIABLE_WIND_CONFIG_lvl3["min_wind_speed"] = 5

VARIABLE_WIND_CONFIG_lvl3 = VARIABLE_WIND_CONFIG.copy()
VARIABLE_WIND_CONFIG_lvl3["max_wind_amplitude"] = 3
VARIABLE_WIND_CONFIG_lvl3["max_wind_speed"] = 25
VARIABLE_WIND_CONFIG_lvl3["min_wind_speed"] = 5
VARIABLE_WIND_CONFIG["wind_noise"] = True

CRAZY_ENV_CONFIG = VARIABLE_WIND_CONFIG.copy()
CRAZY_ENV_CONFIG["action_space_increase"] = 3 # Violation will happen with N/N+1 and with size N-1 outside

SCENARIOS = {
    'ConstantWind-v2': {   
        'entry_point': 'gym_rl_mpc.envs:ConstantWind',
        'config': DEFAULT_CONFIG
    },
    'VariableWind-v2': {
        'entry_point': 'gym_rl_mpc.envs:VariableWind',
        'config': VARIABLE_WIND_CONFIG_lvl3
    },
    'CrazyAgent-v2': {
        'entry_point': 'gym_rl_mpc.envs:CrazyAgent',
        'config': CRAZY_ENV_CONFIG
    }
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        kwargs={'env_config': SCENARIOS[scenario]['config']}
    )