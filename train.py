import gym
import gym_turbine
import os
from time import time
import multiprocessing
import argparse
import numpy as np
import pandas as pd
import json

from gym_turbine import reporting

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


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
    :param report_dir: Path to the folder where the report will be saved.
    :param verbose:
    """

    def __init__(self, report_dir: str, verbose: int = 0):
        super(ReportingCallback, self).__init__(verbose)
        self.report_dir = report_dir
        self.verbose = verbose
        self.reported_episodes = 0

    def _on_step(self) -> bool:
        vec_env = self.training_env

        env_histories = vec_env.get_attr('history')     # TODO: This is a bottleneck for framerate!

        history_empty = max(map(len, env_histories)) == 0

        if not history_empty and env_histories[0]['episode_num'] > self.reported_episodes:
            class Struct(object): pass
            report_env = Struct()
            report_env.history = []
            for env_idx in range(len(env_histories)):
                report_env.history.append(env_histories[env_idx])

            if len(report_env.history) > 0:
                reporting.report(env=report_env, report_dir=self.report_dir)
                self.reported_episodes += 1
                if self.verbose:
                    print("reported episode")


        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        try:
            training_data = pd.read_csv(os.path.join(self.report_dir, 'history_data.csv'))
            reporting.make_summary_file(training_data, self.report_dir)
            if self.verbose:
                print("Made summary file of training")
        except FileNotFoundError as e:
            print('Warning: Could not make summary, File not found ' + str(repr(e)))

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.array(self.training_env.get_attr('cumulative_reward')).mean()
        self.logger.record('custom/cumulative_reward', value)
        return True


if __name__ == '__main__':
    NUM_CPUs = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Number of timesteps to train the agent.',
    )
    parser.add_argument(
        '--note',
        type=str,
        default=None,
        help="Note with additional info about training"
    )
    args = parser.parse_args()

    # Define necessary directories
    EXPERIMENT_ID = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', EXPERIMENT_ID, 'agents')
    os.makedirs(agents_dir, exist_ok=True)
    report_dir = os.path.join('logs', EXPERIMENT_ID, 'training_report')
    tensorboard_log = os.path.join('logs', EXPERIMENT_ID, 'tensorboard')

    # Make environment
    env = make_vec_env('TurbineStab-v0', n_envs=NUM_CPUs, vec_env_cls=SubprocVecEnv)

    # Write note and cofig to Note.txt file
    with open(os.path.join('logs', EXPERIMENT_ID, "Note.txt"), "a") as file_object:
        file_object.write("env_config: " + json.dumps(env.get_attr('config')[0]))
        if args.note:
            file_object.write(args.note)

    # Callback to save model at checkpoints during training
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=agents_dir)
    # Callback to report training to file
    reporting_callback = ReportingCallback(report_dir=report_dir, verbose=1)
    # Callback to report additional values to tensorboard
    tensorboard_callback = TensorboardCallback()
    # Create the callback list
    callback = CallbackList([checkpoint_callback, reporting_callback, tensorboard_callback])

    # Make and train agent
    agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)
    agent.learn(total_timesteps=args.timesteps, callback=callback)

    # Save trained agent
    agents_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps) + ".pkl")
    agent.save(agents_path)

    env.close()
