import numpy as np
import os
from pandas import DataFrame

def format_history(env, lastn=100):

    if lastn > -1:
        relevant_history = env.history[-min(lastn, len(env.history)):]
    else:
        relevant_history = env.history

    episode_nums = np.array([obj['episode_num'] for obj in relevant_history])
    crashes = np.array([obj['crashed'] for obj in relevant_history])
    no_crashes = crashes == 0
    avg_abs_theta = np.array([obj['avg_abs_theta'] for obj in relevant_history])
    rewards = np.array([obj['reward'] for obj in relevant_history])
    timesteps = np.array([obj['timesteps'] for obj in relevant_history])
    durations = np.array([obj['duration'] for obj in relevant_history])

    labels = np.array([r"episode", r"reward", r"crash", r"no_crash", r"theta", r"timesteps", r"duration"])

    episode_nums = episode_nums.reshape((len(relevant_history), 1))
    rewards = rewards.reshape((len(relevant_history), 1))
    crashes = crashes.reshape((len(relevant_history), 1))
    no_crashes = no_crashes.reshape((len(relevant_history), 1))
    avg_abs_theta = avg_abs_theta.reshape((len(relevant_history), 1))
    timesteps = timesteps.reshape((len(relevant_history), 1))
    durations = durations.reshape((len(relevant_history), 1))

    report_data = np.hstack([   episode_nums,
                                rewards,
                                crashes,
                                no_crashes,
                                avg_abs_theta,
                                timesteps,
                                durations
                            ])

    df = DataFrame(report_data, columns=labels)

    return df

def report(env, report_dir, lastn=100):
    try:
        os.makedirs(report_dir, exist_ok=True)

        df = format_history(env, lastn)

        file_path = os.path.join(report_dir, "history_data.csv")
        if not os.path.isfile(file_path):
            df.to_csv(file_path)
        else:
            df.to_csv(file_path, mode='a', header=False)

    except PermissionError as e:
        print('Warning: Report files are open - could not update report: ' + str(repr(e)))
    except OSError as e:
        print('Warning: Ignoring OSError: ' + str(repr(e)))

def make_summary_file(data, report_dir, lastn=50):
    os.makedirs(report_dir, exist_ok=True)

    relevant_data = data[-min(lastn, data.shape[0]):]
    crashes = np.array(relevant_data['crash'])
    no_crashes = crashes == 0
    avg_abs_theta = np.array(relevant_data['theta'])
    rewards = np.array(relevant_data['reward'])
    timesteps = np.array(relevant_data['timesteps'])
    durations = np.array(relevant_data['duration'])

    with open(os.path.join(report_dir, 'summary.txt'), 'w') as f:
        f.write('# TOTAL EPISODES TRAINED: {}\n'.format(data.shape[0]))
        f.write('# PERFORMANCE METRICS (LAST {} EPISODES AVG.)\n'.format(min(lastn, data.shape[0])))
        f.write('{:<30}{:<30}\n'.format('Avg. Reward', rewards.mean()))
        f.write('{:<30}{:<30}\n'.format('Std. Reward', rewards.std()))
        f.write('{:<30}{:<30.2%}\n'.format('Avg. Crashes', crashes.mean()))
        f.write('{:<30}{:<30.2%}\n'.format('No Crashes', no_crashes.mean()))
        f.write('{:<30}{:<30}\n'.format('Avg. absolute theta_r [deg]', avg_abs_theta.mean()*(180/np.pi)))
        f.write('{:<30}{:<30}\n'.format('Avg. Timesteps', timesteps.mean()))
        f.write('{:<30}{:<30}\n'.format('Avg. Duration', durations.mean()))
