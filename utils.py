
import numpy as np

from pandas import DataFrame

def simulate_environment(env, agent, max_time):
    global input_labels, state_labels
    state_labels = [r"x_sg", r"x_sw", r"x_hv", r"theta_r", r"theta_p", r"x_tf", r"x_ts", r"x_1", r"x_2", r"x_3", r"x_4"]
    state_dot_labels = [r"x_sg_dot", r"x_sw_dot", r"x_hv_dot", r"theta_r_dot", r"theta_p_dot", r"x_tf_dot", r"x_ts_dot", r"x_1_dot", r"x_2_dot", r"x_3_dot", r"x_4_dot"]
    input_labels = [r"DVA_1", r"DVA_2", r"DVA_3",  r"DVA_4"]
    reward_labels = [r"reward", r"reward_stab"]
    labels = np.hstack(["time", state_labels, state_dot_labels, input_labels, reward_labels])

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        action, _states = agent.predict(env.observation, deterministic=True)
        _, _, done, _ = env.step(action)

    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    last_reward_stab = np.array(env.episode_history['last_reward_stab']).reshape((env.t_step, 1))
    sim_data = np.hstack([  time,
                            env.episode_history['states'],
                            env.episode_history['states_dot'],
                            env.episode_history['input'],
                            last_reward,
                            last_reward_stab,
                        ])
    df = DataFrame(sim_data, columns=labels)
    return df
