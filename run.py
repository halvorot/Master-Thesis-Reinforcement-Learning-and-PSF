import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_turbine
import os
import argparse

import utils
from stable_baselines3 import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent',
        help='Path to agent .pkl file.',
    )
    args = parser.parse_args()

    agent_path = args.agent
    env = gym.make("TurbineStab-v0")
    agent = PPO.load(agent_path)
    sim_df = utils.simulate_environment(env, agent)
    sim_df.to_csv(r'simdata.csv')
