import os
from gym_turbine.envs import TurbineStab3D
import argparse
import numpy as np

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.utils.test_utils import check_learning_achieved

DEFAULT_CONFIG = {
    "step_size": 0.01,
    "min_reward": -2000,
    "max_t_steps": 100000,
    "crash_angle_condition": 45*(np.pi/180),
    "reward_crash": -1000,
    "n_actuators": 4,
    "n_observations": 2,                        # Pitch and roll angle
    "wind_dir": 0,
    "lambda_reward": 0.8,                       # Parameter for how much focus on stabilization vs power usage
    "sigma_p": 0.2,                             # stddev for pitch reward
    "sigma_r": 0.2,                             # stddev for roll reward
}

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=50)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=0.1)


if __name__ == '__main__':

    args = parser.parse_args()
    ray.init(object_store_memory=1e9)

    register_env("turbine", lambda config: TurbineStab3D(config))
    config = {
        "env": "turbine",  # or "corridor" if registered above
        "env_config": DEFAULT_CONFIG,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "lr": 1e-2,
        "num_workers": 1,  # parallelism
        "framework": "tf",
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
