import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick

def plot_ep_rew_mean(filepaths, labels=None, save=False):
    max_timesteps = 0
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
        max_timesteps = np.maximum(max_timesteps, np.max(X))

    # Add axis labels
    ax.set_xlabel(r'Timestep (in million)')
    ax.set_ylabel(r'Episode Length Mean')
    ax.set_xlim([0,max_timesteps])
    ax.grid(True)
    
    if labels:
        ax.legend()

    if save:
        plt.savefig('plots/ep_len_mean_reward_funcs_10M.pdf', bbox_inches='tight')
    
    plt.show()

def plot_ep_crash_mean(filepaths, labels=None, save=False):
    max_timesteps = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(filepaths)):
        df = pd.read_csv(filepaths[i])
        X = df['Step']/1e6
        Y = df['Value']*100
        if labels:
            ax.plot(X, Y, label=labels[i])
        else:
            ax.plot(X, Y)
        max_timesteps = np.maximum(max_timesteps, np.max(X))

    # Add axis labels
    ax.set_xlabel(r'Timestep (in million)')
    ax.set_ylabel(r'Crash rate mean')
    ax.set_xlim([0,max_timesteps])
    ax.grid(True)

    formater = mtick.PercentFormatter(decimals=0)
    ax.yaxis.set_major_formatter(formater)
    
    if labels:
        ax.legend()

    if save:
        plt.savefig('plots/ep_crash_mean_psf_5M.pdf', bbox_inches='tight')
    
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