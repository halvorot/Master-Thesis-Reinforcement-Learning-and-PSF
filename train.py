import gym
import gym_turbine
import os
from time import time
import multiprocessing
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env


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
        _self.save(os.path.join(agents_dir, "model_" + str(timesteps) + ".pkl"))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Number of timesteps to train the agent.',
    )
    args = parser.parse_args()

    EXPERIMENT_ID = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', 'agents', EXPERIMENT_ID)
    os.makedirs(agents_dir, exist_ok=True)
    tensorboard_log = os.path.join('logs', 'tensorboard')

    # env = gym.make('TurbineStab-v0')
    env = make_vec_env('TurbineStab-v0', n_envs=NUM_CPUs)

    agent = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)
    agent.learn(total_timesteps=args.timesteps, tb_log_name=EXPERIMENT_ID, callback=callback)

    save_path = os.path.join(agents_dir, "last_model_" + str(args.timesteps) + ".pkl")
    agent.save(save_path)

    env.close()
