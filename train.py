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

def callback(_locals, _globals):
    _self = _locals['self']
    timesteps = np.sum(_self.get_env().get_attr('total_t_steps')) + _self.get_env().get_attr('t_step')[0]
    if (timesteps) % 1000 == 0:
        _self.save(os.path.join(agents_dir, "model_" + str(timesteps) + ".zip"))

        vec_env = _self.get_env()

        class Struct(object): pass
        report_env = Struct()
        report_env.history = []
        report_env.config = vec_env.get_attr('config')[0]

        env_histories = vec_env.get_attr('history')
        for episode in range(max(map(len, env_histories))):
            for env_idx in range(len(env_histories)):
                if (episode < len(env_histories[env_idx])):
                    report_env.history.append(env_histories[env_idx][episode])
        if len(report_env.history) > 0:
            reporting.report(env=report_env, report_dir=report_dir)
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
    tensorboard_log = os.path.join('logs', 'tensorboard')

    # env = gym.make('TurbineStab-v0')
    env = make_vec_env('TurbineStab-v0', n_envs=NUM_CPUs)

    agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)
    agent.learn(total_timesteps=args.timesteps, tb_log_name=EXPERIMENT_ID, callback=callback)

    agents_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps) + ".pkl")
    agent.save(agents_path)

    env.close()
