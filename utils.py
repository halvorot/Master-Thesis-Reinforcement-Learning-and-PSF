import control
import slycot
import numpy as np
from numpy.linalg import matrix_rank
from scipy.io import savemat
from scipy.io import loadmat
import gym_rl_mpc.utils.state_space as ss
import gym_rl_mpc.utils.state_space_pendulum as ss_p

from pandas import DataFrame

def simulate_episode(env, agent, max_time, lqr=False):
    global input_labels, state_labels
    state_labels = [r"x_sg", r"x_sw", r"x_hv", r"theta_r", r"theta_p", r"x_1", r"x_2", r"x_3", r"x_4"]
    state_dot_labels = [r"x_sg_dot", r"x_sw_dot", r"x_hv_dot", r"theta_r_dot", r"theta_p_dot", r"x_1_dot", r"x_2_dot", r"x_3_dot", r"x_4_dot"]
    input_labels = [r"Fa_1", r"Fa_2", r"Fa_3",  r"Fa_4"]
    reward_labels = r"reward"
    labels = np.hstack(["time", state_labels, state_dot_labels, input_labels, reward_labels])

    if lqr:
        gamma = env.wind_dir
        A = ss.A(gamma)
        B = ss.B(gamma)
        Q = np.identity(18)
        Q[0, 0] = 1.6e-2/10000
        Q[1, 1] = 17.7
        Q[2, 2] = 599
        Q[3, 3] = 9.23e5
        Q[4, 4] = 1.12e4
        Q[9, 9] = 0.25/100
        Q[10, 10] = 285
        Q[11, 11] = 145
        Q[12, 12] = 100*147
        Q[13, 13] = 4*2.7
        R = np.identity(4)
        # K, S, E = control.lqr(A, B, Q, R)  # NOT WORKING OMP error #15
        # K = loadmat('gym_rl_mpc\\utils\\Ksys.mat')["Ksys_lqr"]     # Use K calculated in code from wiley paper
        K = loadmat('gym_rl_mpc\\utils\\lqr_params.mat')["K"]      # Use K calculated in matlab using own system matrices

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        if lqr:
            action = (-K.dot(env.turbine.state))/ss.max_input
            # action = np.array([0,0,0,0])
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

def simulate_pendulum_episode(env, agent, max_time, lqr=False):
    global input_labels, state_labels
    state_labels = [r"theta", r"x_1", r"x_2"]
    state_dot_labels = [r"theta_dot", r"x_1_dot", r"x_2_dot"]
    input_labels = [r"Fa_1", r"Fa_2"]
    reward_labels = r"reward"
    labels = np.hstack(["time", state_labels, state_dot_labels, input_labels, reward_labels])

    if lqr:
        A = ss_p.A()
        B = ss_p.B()
        Q = np.diag([100, 0, 0])
        R = np.identity(2)
        # K, S, E = control.lqr(A, B, Q, R)  # NOT WORKING OMP error #15
        # K = loadmat('gym_rl_mpc\\utils\\Ksys.mat')["Ksys_lqr"]     # Use K calculated in code from wiley paper
        K = loadmat('gym_rl_mpc\\utils\\lqr_params_pendulum.mat')["K"]      # Use K calculated in matlab using own system matrices

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        if lqr:
            action = (-K.dot(env.turbine.state))/ss.max_input
            # action = np.array([0,0])
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


if __name__ == '__main__':
    A = ss_p.A()
    B = ss_p.B()
    Q = np.diag([100, 0, 0])
    R = np.identity(2)
    mdict = {'A': A, 'B': B, 'Q': Q, 'R': R}
    savemat("lqr_params_pendulum.mat", mdict)
