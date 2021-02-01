
import numpy as np

from pandas import DataFrame

def simulate_environment(env, agent, max_time):
    global input_labels, state_labels
    state_labels = [r"x_sg", r"x_sw", r"x_hv", r"theta_r", r"theta_p", r"x_tf", r"x_ts", r"x_1", r"x_2", r"x_3", r"x_4"]
    input_labels = [r"DVA_1", r"DVA_2", r"DVA_3",  r"DVA_4"]
    reward_labels = [r"reward", r"reward_stab", r"reward_power_use"]
    labels = np.hstack(["time", state_labels, input_labels, reward_labels])

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        action, _states = agent.predict(env.observation, deterministic=True)
        _, _, done, _ = env.step(action)
    time = np.array(env.time).reshape((env.t_step, 1))
    sim_data = np.hstack([time, env.past_states, env.past_actions, env.past_rewards])
    df = DataFrame(sim_data, columns=labels)
    return df
