import numpy as np
from pandas import DataFrame

def simulate_episode(env, agent, max_time):
    global input_labels, state_labels
    state_labels = [r"theta", r"theta_dot", r"omega"]
    input_labels = [r"F_thr", r"blade_pitch"]
    reward_labels = r"reward"
    wind_labels = [r"F_w", r"Q_w"]
    labels = np.hstack(["time", state_labels, input_labels, wind_labels, reward_labels])

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        if agent is not None:
            action, _states = agent.predict(env.observation, deterministic=True)
        else:
            action = np.array([0,0.1])
        _, _, done, _ = env.step(action)

    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    states = env.episode_history['states']
    input = env.episode_history['input']
    wind_force = np.array(env.episode_history['wind_force']).reshape((env.t_step, 1))
    wind_torque = np.array(env.episode_history['wind_torque']).reshape((env.t_step, 1))
    sim_data = np.hstack([  time,
                            states,
                            input,
                            wind_force,
                            wind_torque,
                            last_reward
                        ])
    df = DataFrame(sim_data, columns=labels)
    return df