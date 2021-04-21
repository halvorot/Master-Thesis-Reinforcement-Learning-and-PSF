import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_gen_performance(save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    xlabels = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    level0test = [0,1,1,1,1,1]
    level1test = [1,0,1,1,1,1]
    level2test = [1,1,0,1,1,1]
    level3test = [1,1,1,0,1,1]
    level4test = [1,1,1,1,0,1]
    level5test = [1,1,1,1,1,0]

    x = np.arange(len(xlabels))     # the label locations
    bar_width = 0.1                 # the width of the bars

    r1 = np.arange(len(level0test))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]

    ax.bar(r1, level0test, width=bar_width, edgecolor='white', label='Level 0')
    ax.bar(r2, level1test, width=bar_width, edgecolor='white', label='Level 1')
    ax.bar(r3, level2test, width=bar_width, edgecolor='white', label='Level 2')
    ax.bar(r3, level3test, width=bar_width, edgecolor='white', label='Level 3')
    ax.bar(r3, level4test, width=bar_width, edgecolor='white', label='Level 4')
    ax.bar(r3, level5test, width=bar_width, edgecolor='white', label='Level 5')
    # Add axis labels
    ax.set_xlabel('Agent Train Level', fontweight='bold')
    ax.set_ylabel('Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    if save:
        plt.savefig('plots/ep_rew_mean_10M.pdf', bbox_inches='tight')
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    plot_gen_performance(save=args.save)