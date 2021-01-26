import os
from gym_turbine.envs import TurbineStab3D
import argparse
import numpy as np



DEFAULT_CONFIG = {
    "step_size": 0.01,
    "min_reward": -2000,
    "max_t_steps": 100000,
    "crash_angle_condition": 45*(np.pi/180),
    "max_init_angle": 45*(np.pi/180),
    "reward_crash": -1000,
    "n_actuators": 4,
    "n_observations": 2,                        # Pitch and roll angle
    "wind_dir": 0,
    "lambda_reward": 0.8,                       # Parameter for how much focus on stabilization vs power usage
    "sigma_p": 0.2,                             # stddev for pitch reward
    "sigma_r": 0.2,                             # stddev for roll reward
}



if __name__ == '__main__':

    
