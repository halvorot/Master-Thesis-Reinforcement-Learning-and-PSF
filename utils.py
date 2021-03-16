import numpy as np
from pandas import DataFrame

def simulate_episode(env, agent, max_time):
    global input_labels, state_labels
    state_labels = [r"theta", r"theta_dot", r"omega"]
    input_labels = [r"F_thr", r"blade_pitch", r"power"]
    wind_labels = [r"F_w", r"Q_w", r"Q_gen", r"adjusted_wind_speed"]
    reward_labels = [r"reward", r"theta_reward", r"theta_dot_reward", r"omega_reward", r"power_reward", r"control_reward"]
    
    labels = np.hstack(["time", state_labels, input_labels, wind_labels, reward_labels])

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        if agent is not None:
            action, _states = agent.predict(env.observation, deterministic=True)
        else:
            action = np.array([0,0,0])
        _, _, done, _ = env.step(action)

    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    theta_reward = np.array(env.episode_history['theta_reward']).reshape((env.t_step, 1))
    theta_dot_reward = np.array(env.episode_history['theta_dot_reward']).reshape((env.t_step, 1))
    omega_reward = np.array(env.episode_history['omega_reward']).reshape((env.t_step, 1))
    power_reward = np.array(env.episode_history['power_reward']).reshape((env.t_step, 1))
    control_reward = np.array(env.episode_history['control_reward']).reshape((env.t_step, 1))
    states = env.episode_history['states']
    input = env.episode_history['input']
    wind_force = np.array(env.episode_history['wind_force']).reshape((env.t_step, 1))
    wind_torque = np.array(env.episode_history['wind_torque']).reshape((env.t_step, 1))
    generator_torque = np.array(env.episode_history['generator_torque']).reshape((env.t_step, 1))
    adjusted_wind_speed = np.array(env.episode_history['adjusted_wind_speed']).reshape((env.t_step, 1))
    sim_data = np.hstack([  time,
                            states,
                            input,
                            wind_force,
                            wind_torque,
                            generator_torque,
                            adjusted_wind_speed,
                            last_reward,
                            theta_reward,
                            theta_dot_reward,
                            omega_reward,
                            power_reward,
                            control_reward
                        ])
    df = DataFrame(sim_data, columns=labels)
    return df