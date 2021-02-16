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
    state_labels = [r"x_sg", r"x_sw", r"x_hv", r"theta_r", r"theta_p", r"x_tf", r"x_ts", r"x_1", r"x_2", r"x_3", r"x_4"]
    state_dot_labels = [r"x_sg_dot", r"x_sw_dot", r"x_hv_dot", r"theta_r_dot", r"theta_p_dot", r"x_tf_dot", r"x_ts_dot", r"x_1_dot", r"x_2_dot", r"x_3_dot", r"x_4_dot"]
    input_labels = [r"Fa_1", r"Fa_2", r"Fa_3",  r"Fa_4"]
    reward_labels = r"reward"
    labels = np.hstack(["time", state_labels, state_dot_labels, input_labels, reward_labels])

    if lqr:
        gamma = env.wind_dir
        A = ss.A(gamma)
        B = ss.B(gamma)
        Q = np.identity(22)
        Q[0, 0] = 1.6e-2/10000
        Q[1, 1] = 17.7
        Q[2, 2] = 599
        Q[3, 3] = 9.23e5
        Q[4, 4] = 1.12e4
        Q[5, 5] = 1.44e-2
        Q[6, 6] = 1.56e1
        Q[11, 11] = 0.25/100
        Q[12, 12] = 285
        Q[13, 13] = 145
        Q[14, 14] = 100*147
        Q[15, 15] = 4*2.7
        Q[16, 16] = 40*0.17e11
        Q[17, 17] = 5*0.17e11
        R = np.identity(4)
        # K, S, E = control.lqr(A, B, Q, R) # Not working
        K = loadmat('Ksys.mat')["Ksys"]

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        if lqr:
            action = -K*env.turbine.state
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
    gamma = 0
    A = ss.A(gamma)
    B = ss.B(gamma)
    C = ss.C()
    Q = np.identity(22)
    Q[0, 0] = 1.6e-2/10000
    Q[1, 1] = 17.7
    Q[2, 2] = 599
    Q[3, 3] = 9.23e5
    Q[4, 4] = 1.12e4
    Q[5, 5] = 1.44e-2
    Q[6, 6] = 1.56e1
    Q[11, 11] = 0.25/100
    Q[12, 12] = 285
    Q[13, 13] = 145
    Q[14, 14] = 100*147
    Q[15, 15] = 4*2.7
    Q[16, 16] = 40*0.17e11
    Q[17, 17] = 5*0.17e11
    Q_bar = np.matmul( np.matmul(np.transpose(C), Q), C)
    R = np.identity(4)
    mdict = {"Q": Q, "Q_bar": Q_bar, "R": R, "A": A, "B": B}
    savemat('lqr_params.mat', mdict)
    ctrl = control.ctrb(A, B)
    print(f"Rank ctrb: {matrix_rank(ctrl)}")
    K, S, E = control.lqr(A, B, Q_bar, R)
    print(K.shape)
