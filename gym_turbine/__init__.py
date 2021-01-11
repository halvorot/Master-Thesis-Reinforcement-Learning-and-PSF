from gym.envs.registration import register

DEFAULT_CONFIG = {
    "min_reward": -2000,
    "reward_collision": -1000
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
