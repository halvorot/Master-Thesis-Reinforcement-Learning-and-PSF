import gym
import gym_rl_mpc
import os
from time import time
import multiprocessing
import argparse
import numpy as np
import json

from gym_rl_mpc import reporting

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


hyperparams = {
    'n_steps': 1024,
    'learning_rate': 1e-5,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 4,
    'clip_range': 0.2,
    'ent_coef': 0.01,
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

    def _on_step(self) -> bool:
        # check if env is done, if yes report it to csv file
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))
        if np.sum(done_array).item() > 0:
            env_histories = self.training_env.get_attr('history')

            class Struct(object): pass
            report_env = Struct()
            report_env.history = []
            for env_idx in range(len(done_array)):
                if done_array[env_idx]:
                    report_env.history.append(env_histories[env_idx])

            reporting.report(env=report_env, report_dir=self.report_dir)


        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        vec_env = self.training_env
        env_histories = vec_env.get_attr('total_history')
        class Struct(object): pass
        report_env = Struct()
        report_env.history = []
        for episode in range(max(map(len, env_histories))):
            for env_idx in range(len(env_histories)):
                if (episode < len(env_histories[env_idx])):
                    report_env.history.append(env_histories[env_idx][episode])

        if len(report_env.history) > 0:
            training_data = reporting.format_history(report_env, lastn=100)
            reporting.make_summary_file(training_data, self.report_dir, len(report_env.history))
            if self.verbose:
                print("Made summary file of training")

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        self.start_time = time()
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        done_array = np.array(self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones"))

        if np.sum(done_array).item():
            history = self.training_env.get_attr('history')

            for env_idx in range(len(done_array)):
                if done_array[env_idx]:
                    self.logger.record_mean('custom/crashed', history[env_idx]['crashed'])
                    self.logger.record_mean('custom/wind_speed', history[env_idx]['wind_speed'])
                    self.logger.record_mean('custom/theta_reward', history[env_idx]['theta_reward'])
                    self.logger.record_mean('custom/theta_dot_reward', history[env_idx]['theta_dot_reward'])
                    self.logger.record_mean('custom/omega_reward', history[env_idx]['omega_reward'])
                    self.logger.record_mean('custom/power_reward', history[env_idx]['power_reward'])
                    self.logger.record_mean('custom/input_reward', history[env_idx]['input_reward'])
                    self.logger.record_mean('custom/psf_reward', history[env_idx]['psf_reward'])
        
        self.logger.record("time/custom_time_elapsed", int(time() - self.start_time))

        return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Number of timesteps to train the agent. Default=1000000',
    )
    parser.add_argument(
        '--agent',
        help='Path to the RL agent to continue training from.',
    )
    parser.add_argument(
        '--env',
        type=str,
        default='VariableWind-v1',
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
    parser.add_argument(
        '--note',
        type=str,
        default=None,
        help="Note with additional info about training"
    )
    parser.add_argument(
        '--no_reporting',
        help='Skip reporting to increase framerate',
        action='store_true'
    )
    parser.add_argument(
        '--psf',
        help='Use psf corrected action',
        action='store_true'
    )
    parser.add_argument(
        '--single_cpu',
        help='Use only one CPU (no parallelization)',
        action='store_true'
    )
    args = parser.parse_args()

    NUM_CPUs = multiprocessing.cpu_count() if args.single_cpu == False else 1

    # Make environment (NUM_CPUs parallel envs)
    env_id = args.env
    if args.psf:
        customconfig = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
        customconfig['use_psf'] = True
        env_kwargs = {'env_config': customconfig}
        print("Using PSF corrected actions")
    else:
        env_kwargs = None

    env = make_vec_env(env_id, n_envs=NUM_CPUs, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    

    # Define necessary directories
    EXPERIMENT_ID = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', env_id, EXPERIMENT_ID, 'agents')
    os.makedirs(agents_dir, exist_ok=True)
    report_dir = os.path.join('logs', env_id, EXPERIMENT_ID, 'training_report')
    tensorboard_log = os.path.join('logs', env_id, EXPERIMENT_ID, 'tensorboard')

    # Write note and config to Note.txt file
    with open(os.path.join('logs', env_id, EXPERIMENT_ID, "Note.txt"), "a") as file_object:
        file_object.write("env_config: " + json.dumps(env.get_attr('config')[0]) + "\n")
        file_object.write("hyperparams: " + json.dumps(hyperparams) + "\n")
        if args.note:
            file_object.write(args.note)

    # Callback to save model at checkpoints during training
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=agents_dir)
    # Callback to report training to file
    reporting_callback = ReportingCallback(report_dir=report_dir, verbose=True)
    # Callback to report additional values to tensorboard
    tensorboard_callback = TensorboardCallback(verbose=True)
    # Create the callback list
    if args.no_reporting:
        callback = CallbackList([checkpoint_callback, tensorboard_callback])
    else:
        callback = CallbackList([checkpoint_callback, reporting_callback, tensorboard_callback])

    if (args.agent is not None):
        agent = PPO.load(args.agent, env=env, verbose=True, tensorboard_log=tensorboard_log)
    else:
        agent = PPO('MlpPolicy', env, verbose=True, tensorboard_log=tensorboard_log, **hyperparams)

    agent.learn(total_timesteps=args.timesteps, callback=callback)

    # Save trained agent
    agent_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps))
    agent.save(agent_path)

    env.close()
