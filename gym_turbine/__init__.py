from gym.envs.registration import register
import numpy as np

DEFAULT_CONFIG = {
    "min_reward": -2000,
    "max_t_steps": 10000,
    "crash_angle_condition": 30*(np.pi/180),
    "reward_crash": -1000
}

SCENARIOS = {
    'TestScenario-v0': {
        'entry_point': 'gym_turbine.envs:TestScenario',
        'config': DEFAULT_CONFIG
    },
}

for scenario in SCENARIOS:
    register(
        id=scenario,
        entry_point=SCENARIOS[scenario]['entry_point'],
        kwargs={'env_config': SCENARIOS[scenario]['config']}
    )
