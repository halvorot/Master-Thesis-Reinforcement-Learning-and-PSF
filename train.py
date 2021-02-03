import gym
import gym_turbine
import os
from time import time
import multiprocessing
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


NUM_CPUs = multiprocessing.cpu_count()

hyperparams = {
    'n_steps': 1024,
    'nminibatches': 256,
    'learning_rate': 1e-5,
    'lam': 0.95,
    'gamma': 0.99,
    'noptepochs': 4,
    'cliprange': 0.2,
    'ent_coef': 0.01,
    'verbose': 2
}

class CustomCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps
    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Number of timesteps to train the agent.',
    )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=1000,
        help='How often to save a checkpoint of the agent. Every save_freq steps.',
    )
    args = parser.parse_args()

    EXPERIMENT_ID = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', 'agents', EXPERIMENT_ID)
    tensorboard_log = os.path.join('logs', 'tensorboard')
    custom_callback = CustomCallback(save_freq=args.save_freq, save_path=agents_dir)

    # env = gym.make('TurbineStab-v0')
    env = make_vec_env('TurbineStab-v0', n_envs=NUM_CPUs)

    agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)
    agent.learn(total_timesteps=args.timesteps, tb_log_name=EXPERIMENT_ID, callback=custom_callback)

    save_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps) + ".pkl")
    agent.save(save_path)

    env.close()
