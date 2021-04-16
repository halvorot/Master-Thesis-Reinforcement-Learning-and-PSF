import argparse
import os

import gym
import numpy as np
from pandas.core.frame import DataFrame
from stable_baselines3 import PPO

import gym_rl_mpc
import utils
from gym_rl_mpc.utils import model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        required=True,
        choices=gym_rl_mpc.SCENARIOS.keys(),
        help="Environment to run."
    )
    parser.add_argument(
        '--agent',
        required=True,
        help='Path to agent .zip file.',
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        required=True,
        default=100,
        help='Number of episodes to simulate.',
    )
    parser.add_argument(
        '--time',
        type=int,
        default=300,
        help='Max simulation time (seconds).',
    )
    parser.add_argument(
        '--psf',
        help='Use psf corrected action',
        action='store_true'
    )
    args = parser.parse_args()
    return args


def test(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    if args.psf:
        config = gym_rl_mpc.SCENARIOS[args.env]['config'].copy()
        config['use_psf'] = True
        print("Using PSF corrected actions")
    else:
        config = gym_rl_mpc.SCENARIOS[args.env]['config']
    env = gym.make(args.env, env_config=config)
    env_id = env.unwrapped.spec.id

    agent_path = args.agent
    agent = PPO.load(agent_path)
    cumulative_rewards = []
    crashes = []
    print(f'Testing in "{env_id}" for {args.num_episodes} episodes.\n ')
    report_msg_header = '{:<20}{:<20}{:<20}{:<20}'.format('Episode', 'Timesteps', 'Cum. Reward', 'Progress')
    print(report_msg_header)
    print('-'*len(report_msg_header)) 
    for episode in range(args.num_episodes):
        sim_df = utils.simulate_episode(env=env, agent=agent, max_time=args.time, verbose=True, id=f'Episode {episode}')
        print(env.cumulative_reward)
        cumulative_rewards.append(np.sum(sim_df['reward']))
        crashes.append(env.crashed)
        print('')
    print(f'Avg. Cumulative reward: {np.sum(cumulative_rewards)/args.num_episodes}')

    test_df = DataFrame(cumulative_rewards, columns=['cumulative_reward'])
    
    agent_path_list = agent_path.split("\\")
    testdata_dir = os.path.join("logs", agent_path_list[-4], agent_path_list[-3], "test_data")
    os.makedirs(testdata_dir, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(testdata_dir, env_id + "_" + agent_path_list[-1][0:-4] + f"_testdata_{i}.csv")):
        i += 1
    test_df.to_csv(os.path.join(testdata_dir, env_id + "_" + agent_path_list[-1][0:-4] + f"_testdata_{i}.csv"))
    print(f'Reported to file in: {testdata_dir}')
    env.close()


if __name__ == "__main__":
    args = parse_argument()
    test(args)
