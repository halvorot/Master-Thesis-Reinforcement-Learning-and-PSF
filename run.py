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
        help='Path to agent .pkl file or model .zip file.',
        required=True,
    )
    parser.add_argument(
        '--time',
        type=int,
        default=50,
        help='Max simulation time (seconds).',
    )
    args = parser.parse_args()

    agent_path = args.agent
    env = gym.make("TurbineStab-v0")
    agent = PPO.load(agent_path)
    sim_df = utils.simulate_environment(env, agent, args.time)

    agent_path_list = agent_path.split("\\")
    simdata_dir = os.path.join("logs", agent_path_list[-3], "sim_data")
    os.makedirs(simdata_dir, exist_ok=True)

    # Save file to logs\<EXPERIMENT_ID>\sim_data\<agent_file_name>_simdata.csv
    i = 0
    while os.path.exists(os.path.join(simdata_dir, agent_path_list[-1][0:-4] + f"_simdata_{i}.csv")):
        i += 1
    sim_df.to_csv(os.path.join(simdata_dir, agent_path_list[-1][0:-4] + f"_simdata_{i}.csv"))
