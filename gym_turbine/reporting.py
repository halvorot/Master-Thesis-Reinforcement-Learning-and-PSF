import numpy as np
import os
from pandas import DataFrame


def report(env, report_dir, lastn=100):
    try:
        os.makedirs(report_dir, exist_ok=True)

        if lastn > -1:
            relevant_history = env.history[-min(lastn, len(env.history)):]
        else:
            relevant_history = env.history

        episode_nums = np.array([obj['episode_num'] for obj in relevant_history])
        crashes = np.array([obj['crashed'] for obj in relevant_history])
        no_crashes = crashes == 0
        avg_x_tf = np.array([obj['avg_x_tf'] for obj in relevant_history])
        avg_x_ts = np.array([obj['avg_x_ts'] for obj in relevant_history])
        avg_theta_r = np.array([obj['avg_theta_r'] for obj in relevant_history])
        avg_theta_p = np.array([obj['avg_theta_p'] for obj in relevant_history])
        rewards = np.array([obj['reward'] for obj in relevant_history])
        timesteps = np.array([obj['timesteps'] for obj in relevant_history])
        durations = np.array([obj['duration'] for obj in relevant_history])


        with open(os.path.join(report_dir, 'report.txt'), 'w') as f:
            f.write('# PERFORMANCE METRICS (LAST {} EPISODES AVG.)\n'.format(min(lastn, len(env.history))))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Reward', rewards.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Std. Reward', rewards.std()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Crashes', crashes.mean()))
            f.write('{:<30}{:<30.2%}\n'.format('No Crashes', no_crashes.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. x_tf', avg_x_tf.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. x_ts', avg_x_ts.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. theta_r', avg_theta_r.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. theta_p', avg_theta_p.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Timesteps', timesteps.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Duration', durations.mean()))

        labels = np.array([r"episode", r"reward", r"crash", r"no_crash", r"x_tf", r"x_ts", r"theta_r", r"theta_p", r"timesteps", r"duration"])

        episode_nums = episode_nums.reshape((len(relevant_history), 1))
        rewards = rewards.reshape((len(relevant_history), 1))
        crashes = crashes.reshape((len(relevant_history), 1))
        no_crashes = no_crashes.reshape((len(relevant_history), 1))
        avg_x_tf = avg_x_tf.reshape((len(relevant_history), 1))
        avg_x_ts = avg_x_ts.reshape((len(relevant_history), 1))
        avg_theta_r = avg_theta_r.reshape((len(relevant_history), 1))
        avg_theta_p = avg_theta_p.reshape((len(relevant_history), 1))
        timesteps = timesteps.reshape((len(relevant_history), 1))
        durations = durations.reshape((len(relevant_history), 1))

        report_data = np.hstack([   episode_nums,
                                    rewards,
                                    crashes,
                                    no_crashes,
                                    avg_x_tf,
                                    avg_x_ts,
                                    avg_theta_r,
                                    avg_theta_p,
                                    timesteps,
                                    durations
                                ])

        df = DataFrame(report_data, columns=labels)

        df.to_csv(os.path.join(report_dir, "history_data.csv"))

    except PermissionError as e:
        print('Warning: Report files are open - could not update report: ' + str(repr(e)))
    except OSError as e:
        print('Warning: Ignoring OSError: ' + str(repr(e)))
