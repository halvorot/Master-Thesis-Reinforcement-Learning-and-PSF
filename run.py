import gym
import gym_rl_mpc
import os
import argparse

import utils
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--agent',
        help='Path to agent .zip file.',
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

    env = gym.make("PendulumStab-v1")
    env_id = env.unwrapped.spec.id
    if args.agent:
        agent_path = args.agent
        agent = PPO.load(agent_path)
        sim_df = utils.simulate_episode(env=env, agent=agent, max_time=args.time)
        agent_path_list = agent_path.split("\\")
        simdata_dir = os.path.join("logs", env_id, agent_path_list[-3], "sim_data")
        os.makedirs(simdata_dir, exist_ok=True)

        # Save file to logs\env_id\<EXPERIMENT_ID>\sim_data\<agent_file_name>_simdata.csv
        i = 0
        while os.path.exists(os.path.join(simdata_dir, agent_path_list[-1][0:-4] + f"_simdata_{i}.csv")):
            i += 1
        sim_df.to_csv(os.path.join(simdata_dir, agent_path_list[-1][0:-4] + f"_simdata_{i}.csv"))
    else:
        sim_df = utils.simulate_episode(env=env, agent=None, max_time=args.time)

    env.close()

    if args.plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        time = np.array(range(0, len(sim_df['theta'])))*env.step_size

        ax1.plot(time, sim_df['theta']*(180/np.pi), label='theta')
        ax1.plot(time, np.zeros(len(time)), linestyle='--', color='k')
        ax1.set_ylabel('Degrees')
        ax1.set_title('platform_angle')
        ax1.legend()

        ax2.plot(time, sim_df['omega']*(180/np.pi), label='omega')
        ax2.set_ylabel('Degrees/sec')
        ax2.set_title('Angluar velocity turbine')
        ax2.legend()

        ax3.plot(time, sim_df['F_thr'], label='F_thr')
        ax3.plot(time, sim_df['blade_pitch'], label='Blade pitch')
        ax3.set_ylabel('[N]')
        ax3.set_title('Input')
        ax3.legend()

        ax4.plot(time, sim_df['F_w'], label='F_w')
        ax4.plot(time, sim_df['Q_w'], label='Q_w')
        ax4.set_ylabel('[N], [Nm]')
        ax4.set_title('Wind')
        ax4.legend()

        plt.show()
