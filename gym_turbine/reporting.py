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
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Timesteps', timesteps.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Duration', durations.mean()))

        data_labels = [r"reward", r"crash", r"no_crash", r"x_tf", r"x_ts", r"timesteps", r"duration"]
        labels = np.hstack(["episode", data_labels])

        report_data = np.hstack([   episode_nums,
                                    rewards,
                                    crashes,
                                    no_crashes,
                                    avg_x_tf,
                                    avg_x_ts,
                                    timesteps,
                                    durations
                                ])

        df = DataFrame(report_data, columns=labels)

        df.to_csv(os.path.join(report_dir, "history_data.csv"))

    except PermissionError as e:
        print('Warning: Report files are open - could not update report: ' + str(repr(e)))
    except OSError as e:
        print('Warning: Ignoring OSError: ' + str(repr(e)))
