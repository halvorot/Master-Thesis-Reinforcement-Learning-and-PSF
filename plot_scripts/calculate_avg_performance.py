import argparse
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--filename',
    type=str,
    required=True,
    help="Filename"
)
args = parser.parse_args()

df = pd.read_csv(args.filename)
crashes = df['crash'].values
performances = df['performance'].values
crash_indexes = []
for i in range(len(crashes)):
    if crashes[i]:
        crash_indexes.append(i)
performances_no_crash = np.delete(performances,crash_indexes)
print(f'Avg. Performance: {np.sum(performances_no_crash)/(len(crashes)-np.sum(crashes))}')
print(f'Crashes: {np.sum(crashes)*100/len(crashes)}%')