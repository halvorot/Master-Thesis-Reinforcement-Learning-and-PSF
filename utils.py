import control
import slycot
import numpy as np
from numpy.linalg import matrix_rank
from scipy.io import savemat
from scipy.io import loadmat
import gym_turbine.utils.state_space as ss

from pandas import DataFrame

def simulate_episode(env, agent, max_time, lqr=False):
    global input_labels, state_labels
    state_labels = [r"x_sg", r"x_sw", r"x_hv", r"theta_r", r"theta_p", r"x_1", r"x_2", r"x_3", r"x_4"]
    state_dot_labels = [r"x_sg_dot", r"x_sw_dot", r"x_hv_dot", r"theta_r_dot", r"theta_p_dot", r"x_1_dot", r"x_2_dot", r"x_3_dot", r"x_4_dot"]
    input_labels = [r"Fa_1", r"Fa_2", r"Fa_3",  r"Fa_4"]
    reward_labels = r"reward"
    labels = np.hstack(["time", state_labels, state_dot_labels, input_labels, reward_labels])

    if lqr:
        K = loadmat('gym_turbine\\utils\\Ksys.mat')["Ksys_lqr"]     # Use K calculated in code from wiley paper
        # K = loadmat('gym_turbine\\utils\\LQR_params.mat')["K"]      # Use K calculated in matlab using own system matrices

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        if lqr:
            action = (-K.dot(env.turbine.state))/ss.max_input
        else:
            action, _states = agent.predict(env.observation, deterministic=True)
        _, _, done, _ = env.step(action)

    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    sim_data = np.hstack([  time,
                            env.episode_history['states'],
                            env.episode_history['states_dot'],
                            env.episode_history['input'],
                            last_reward,
                        ])
    df = DataFrame(sim_data, columns=labels)
    return df
