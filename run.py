import gym
import gym_rl_mpc
import os
import argparse

import utils
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from gym_rl_mpc.utils import model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM, RPM2RAD, DEG2RAD


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='VariableWind-v0',
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
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
    parser.add_argument(
        '--psf',
        help='Use psf corrected action',
        action='store_true'
    )
    args = parser.parse_args()

    if args.psf:
        config = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
        config['use_psf'] = True
        print("Using PSF corrected actions")
    else:
        config = gym_rl_mpc.SCENARIOS[args.env]['config']
    env = gym.make(args.env, env_config=config)
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
        if args.env == 'ConstantWind-v0':
            fig.suptitle(f"Wind speed: {env.wind_speed:.1f} m/s")
        else:
            fig.suptitle(f"Wind mean: {env.wind_mean:.1f} m/s, Wind amplitude: {env.wind_amplitude:.1f} m/s")

        time = np.array(range(0, len(sim_df['theta'])))*env.step_size

        ax1.plot(time, sim_df['theta']*RAD2DEG, label='$\\theta$')
        ax1.plot(time, sim_df['theta_dot']*RAD2DEG, label='$\dot{\\theta}$')
        ax1.plot(time, np.zeros(len(time)), linestyle='--', color='k')
        ax1.set_ylabel('Degrees, deg/sec')
        ax1.set_title('platform angle and angular velocity')
        ax1.legend()

        ax2.plot(time, sim_df['omega']*RAD2RPM, label='$\Omega$')
        ax2.set_ylabel('rpm')
        ax2.set_title('Angluar Velocity Rotor')
        ax2.legend()

        color = 'tab:blue'
        ax3.plot(time, sim_df['F_thr'], label='F_thr', color=color)
        ax3.set_ylabel('F_thr [N]', color=color)
        ax3.set_title('Input')
        ax3.legend()

        color = 'tab:orange'
        ax3_2 = ax3.twinx()
        ax3_2.plot(time, sim_df['blade_pitch']*RAD2DEG, label='Blade pitch', color=color)
        ax3_2.set_ylabel('Blade pitch [Degrees]', color=color)
        ax3_2.legend()

        color = 'tab:blue'
        ax4.plot(time, sim_df['F_w'], label='F_w', color=color)
        ax4.plot(time, np.zeros(len(time)), linestyle='--', color='k', linewidth=0.5)
        ax4.set_ylabel('F_w [N]', color=color)
        ax4.set_title('Wind')
        ax4.legend()

        color = 'tab:orange'
        ax4_2 = ax4.twinx()
        ax4_2.plot(time, sim_df['Q_w'], label='Q_w', color=color)
        ax4_2.plot(time, sim_df['Q_gen'], label='Q_gen', color='tab:red')
        ax4_2.set_ylabel('Q [Nm]', color=color)
        ax4_2.legend()


        fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2)
        ax21.plot(time, sim_df['agent_F_thr']*params.max_thrust_force, label='agent thrust')
        ax21.plot(time, sim_df['psf_F_thr']*params.max_thrust_force, label='agent thrust')
        ax21.set_ylabel('F_thr [N]')
        ax21.set_title('Commanded Thrust Force')
        ax21.legend()

        ax22.plot(time, np.array(sim_df['agent_power'])*params.max_power_generation, label='agent power')
        ax22.plot(time, np.array(sim_df['psf_power'])*params.max_power_generation, label='PSF power')
        ax22.set_ylabel('Power')
        ax22.set_title('Commanded Power')
        ax22.legend()

        ax23.plot(time, np.array(sim_df['agent_blade_pitch'])*params.max_blade_pitch*RAD2DEG, label='agent blade pitch')
        ax23.plot(time, np.array(sim_df['psf_blade_pitch'])*params.max_blade_pitch*RAD2DEG, label='PSF blade pitch')
        ax23.set_ylabel('Blade pitch [deg]')
        ax23.set_title('Commanded blade pitch')
        ax23.legend()

        ax24.plot(time, np.array(sim_df['wind_speed']), label='Wind speed')
        ax24.set_ylabel('Wind speed [m/s]')
        ax24.set_title('Wind')
        ax24.legend()

        plt.show()
