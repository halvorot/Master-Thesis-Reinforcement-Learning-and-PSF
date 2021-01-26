import gym
import gym_turbine
import numpy as np
import os
from time import time

from stable_baselines3 import PPO

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
    global n_steps, best_mean_reward
    if (n_steps + 1) % 1000 == 0:
        _locals['self'].save(os.path.join(agents_dir, "model_" + str(n_steps+1) + ".pkl"))
    n_steps += 1
    return True


if __name__ == '__main__':
    EXPERIMENT_ID = str(int(time())) + 'ppo'
    agents_dir = os.path.join('logs', 'agents', EXPERIMENT_ID)
    os.makedirs(agents_dir, exist_ok=True)
    tensorboard_dir = os.path.join('logs', 'tensorboard', 'TurbineEnv', EXPERIMENT_ID)
    tensorboard_port = 6006
    hyperparams["tensorboard_log"] = tensorboard_dir
    n_steps = 0

    env = gym.make('TurbineStab-v0')

    agent = PPO('MlpPolicy', env, verbose=1)
    agent.learn(total_timesteps=10000, tb_log_name="PPO", callback=callback)

    save_path = os.path.join(agents_dir, "last_model.pkl")
    agent.save(save_path)

    env.close()
