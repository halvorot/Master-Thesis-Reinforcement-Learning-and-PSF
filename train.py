import gym
import gym_turbine
import os
from time import time
import multiprocessing
import argparse
import numpy as np

from gym_turbine import reporting

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback


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

class ReportingCallback(BaseCallback):
    """
    Callback for reporting training
    :param save_freq:
    :param report_dir: Path to the folder where the report will be saved.
    :param verbose:
    """

    def __init__(self, save_freq: str, report_dir: str, verbose: int = 0):
        super(ReportingCallback, self).__init__(verbose)
        self.report_dir = report_dir
        self.save_freq = save_freq
        self.verbose = verbose

    def _on_step(self) -> bool:
        vec_env = self.training_env

        class Struct(object): pass
        report_env = Struct()
        report_env.history = []
        report_env.config = vec_env.get_attr('config')[0]

        env_histories = vec_env.get_attr('history')
        for episode in range(max(map(len, env_histories))):
            for env_idx in range(len(env_histories)):
                if (episode < len(env_histories[env_idx])):
                    report_env.history.append(env_histories[env_idx][episode])
        if len(report_env.history) > 0 and self.n_calls % self.save_freq == 0:
            reporting.report(env=report_env, report_dir=self.report_dir)
            if self.verbose:
                print("reporting...")
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
    os.makedirs(agents_dir, exist_ok=True)
    report_dir = os.path.join('logs', 'reports', EXPERIMENT_ID)
    results_dir = os.path.join('logs', 'results', EXPERIMENT_ID)
    tensorboard_log = os.path.join('logs', 'tensorboard')

    # Make environment
    env = make_vec_env('TurbineStab-v0', n_envs=NUM_CPUs, vec_env_cls=SubprocVecEnv)

    # Callback to save model at checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=agents_dir)
    # Callback to report training to file
    reporting_callback = ReportingCallback(save_freq=10, report_dir=report_dir, verbose=1)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, reporting_callback])

    agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)
    agent.learn(total_timesteps=args.timesteps, tb_log_name=EXPERIMENT_ID, callback=callback)

    agents_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps) + ".pkl")
    agent.save(agents_path)

    env.close()
