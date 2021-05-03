import argparse
import numpy as np
import pandas as pd
import argparse



def calculate_avg_performance(filename):
    df = pd.read_csv(filename)
    crashes = df['crash'].values
    performances = df['performance'].values
    crash_indexes = []
    for i in range(len(crashes)):
        if crashes[i]:
            crash_indexes.append(i)
    performances_no_crash = np.delete(performances,crash_indexes)
    avg_perf = np.sum(performances_no_crash)/(len(crashes)-np.sum(crashes))
    avg_crash = np.sum(crashes)*100/len(crashes)
    return avg_perf, avg_crash

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        type=str,
        required=True,
        help="Filename"
    )
    args = parser.parse_args()
    avg_performance, avg_crashes = calculate_avg_performance(args.filename)
    print(f'Avg. Performance: {avg_performance}')
    print(f'Crashes: {avg_crashes}%')