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

    env = gym.make("PendulumStab-v0")
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
        recorded_states = []
        recorded_inputs = []
        recorded_disturbance = []
        state_labels = [r"theta", r"theta_dot"]
        input_labels = [r"F"]
        disturbance_labels = [r"F_d"]
        labels = np.hstack([state_labels, input_labels, disturbance_labels])

        env = gym.make("PendulumStab-v0")
        env_id = env.unwrapped.spec.id
        env.reset()
        env.pendulum.state[0] = 5*(np.pi/180)
        recorded_states.append(env.pendulum.state)
        recorded_inputs.append(env.pendulum.input)
        recorded_disturbance.append(env.pendulum.disturbance_force)
        done = False
        while not done and env.t_step < args.time/env.step_size:
            action = 0
            _, _, done, _ = env.step(action)
            recorded_states.append(env.pendulum.state)
            recorded_inputs.append(env.pendulum.input)
            recorded_disturbance.append(env.pendulum.disturbance_force)
        
        recorded_inputs = np.array(recorded_inputs).reshape((len(recorded_inputs), 1))
        recorded_disturbance = np.array(recorded_disturbance).reshape((len(recorded_disturbance), 1))

        sim_data = np.hstack([  recorded_states,
                                recorded_inputs,
                                recorded_disturbance,
                            ])
        sim_df = DataFrame(sim_data, columns=labels)

    env.close()

    if args.plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        time = np.array(range(0, len(sim_df['theta'])))*env.step_size

        ax1.plot(time, sim_df['theta']*(180/np.pi), label='theta')
        ax1.set_ylabel('Degrees')
        ax1.set_title('platform_angle')
        ax1.legend()

        ax2.plot(time, sim_df['theta_dot']*(180/np.pi), label='theta_dot')
        ax2.set_ylabel('Degrees/sec')
        ax2.set_title('Angluar velocity')
        ax2.legend()

        ax3.plot(time, sim_df['F'], label='F')
        ax3.set_ylabel('[N]')
        ax3.set_title('Input')
        ax3.legend()

        ax4.plot(time, sim_df['F_d'], label='F_d')
        ax4.set_ylabel('[N]')
        ax4.set_title('Disturbance force')
        ax4.legend()

        plt.show()
