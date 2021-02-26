import numpy as np
from pandas import DataFrame

def simulate_episode(env, agent, max_time, lqr=False):
    global input_labels, state_labels
    state_labels = [r"theta", r"theta_dot", r"x_1", r"x_1_dot", r"x_2", r"x_2_dot"]
    input_labels = [r"Fa_1", r"Fa_2"]
    reward_labels = r"reward"
    labels = np.hstack(["time", state_labels, input_labels, reward_labels])

    done = False
    env.reset()
    while not done and env.t_step < max_time/env.step_size:
        action, _states = agent.predict(env.observation, deterministic=True)
        _, _, done, _ = env.step(action)

    time = np.array(env.episode_history['time']).reshape((env.t_step, 1))
    last_reward = np.array(env.episode_history['last_reward']).reshape((env.t_step, 1))
    sim_data = np.hstack([  time,
                            env.episode_history['states'],
                            env.episode_history['input'],
                            last_reward,
                        ])
    df = DataFrame(sim_data, columns=labels)
    return df