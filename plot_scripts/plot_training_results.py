import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_ep_rew_mean(filepaths, labels=None, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(filepaths)):
        df = pd.read_csv(filepaths[i])
        X = df['Step']/1e6
        Y = df['Value']
        if labels:
            ax.plot(X, Y, label=labels[i])
        else:
            ax.plot(X, Y)

    # Add axis labels
    ax.set_xlabel(r'Timestep (in million)')
    ax.set_ylabel(r'Episode Reward Mean')
    ax.set_xlim([0,np.max(X)])
    ax.grid(True)
    
    if labels:
        ax.legend()

    if save:
        plt.savefig('plots/ep_rew_mean_10M.pdf', bbox_inches='tight')
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        nargs='+',
        help='Path to the CSV file.',
        type=str,
        required=True
    )
    parser.add_argument(
        '--label',
        nargs='+',
        help='Labels for each file',
        type=str,
    )
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    if args.label:
        assert len(args.file)==len(args.label)

    plot_ep_rew_mean(args.file, labels=args.label, save=args.save)