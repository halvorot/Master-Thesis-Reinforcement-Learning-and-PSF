import gym
import gym_rl_mpc
import os
import argparse

import utils
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent',
        help='Path to agent .zip file.',
        required=True
    )
    parser.add_argument(
        '--time',
        type=int,
        default=100,
        help='Max simulation time (seconds).',
    )
    parser.add_argument(
        '--plot',
        help='Show plot of states after simulation',
        action='store_true'
    )
    args = parser.parse_args()

    env = gym.make("TurbineStab-v0")
    env_id = env.unwrapped.spec.id

    agent_path = args.agent
    agent = PPO.load(agent_path)
    sim_df = utils.simulate_episode(env=env, agent=agent, max_time=args.time, lqr=False)
    agent_path_list = agent_path.split("\\")
    simdata_dir = os.path.join("logs", env_id, agent_path_list[-3], "sim_data")
    os.makedirs(simdata_dir, exist_ok=True)

    # Save file to logs\env_id\<EXPERIMENT_ID>\sim_data\<agent_file_name>_simdata.csv
    i = 0
    while os.path.exists(os.path.join(simdata_dir, agent_path_list[-1][0:-4] + f"_simdata_{i}.csv")):
        i += 1
    sim_df.to_csv(os.path.join(simdata_dir, agent_path_list[-1][0:-4] + f"_simdata_{i}.csv"))

    env.close()

    if args.plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        ax1.plot(sim_df['theta']*(180/np.pi), label='theta')
        ax1.set_ylabel('Degrees')
        ax1.set_title('Angle')
        ax1.legend()

        ax2.plot(sim_df['x_1'], label='x_1')
        ax2.plot(sim_df['x_2'], label='x_2')
        ax2.set_ylabel('Meters')
        ax2.set_title('DVA displacements')
        ax2.legend()

        ax3.plot(sim_df['F_1'], label='F_1')
        ax3.plot(sim_df['F_2'], label='F_2')
        ax3.set_ylabel('[N]')
        ax3.set_title('Inputs')
        ax3.legend()

        ax4.plot(sim_df['x_1_dot'], label='x_1_dot')
        ax4.plot(sim_df['x_2_dot'], label='x_2_dot')
        ax4.set_ylabel('m/s')
        ax4.set_title('DVA velocities')
        ax4.legend()

        plt.show()
