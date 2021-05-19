import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot_wind(wind_mean, wind_amplitude):
    t = np.arange(0,300,1)
    # wind_amplitude = min((max_wind_speed - min_wind_speed) / 2, max_wind_amplitude) * np.random.random()
    # wind_mean = (max_wind_speed - min_wind_speed - 2 * wind_amplitude) * np.random.random() + min_wind_speed + wind_amplitude
    wind_phase_shift = 0

    wind_speed = wind_amplitude * np.sin((1 / 60) * 2 * np.pi * t + wind_phase_shift) + wind_mean
    return t, wind_speed
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_wind_speed = 10
    max_wind_speed = 20
    sim_ampl = [1,2,3]
    sim_mean = [11,18,15]
    for i in range(len(sim_ampl)):
        x, y = plot_wind(sim_mean[i],sim_ampl[i])

        ax.plot(x, y)
    x, y = plot_wind(1,1)
    ax.plot(x,[10]*len(x), linestyle='--', color='black')
    ax.plot(x,[20]*len(x), linestyle='--', color='black')

    # Add axis labels
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'w$_0$ [m/s]')
    ax.set_xlim([0,300])
    ax.set_ylim([5, 25])
    ax.grid(True)

    if args.save:
        plt.savefig('plots/wind_sim_10_20.pdf', bbox_inches='tight')

    plt.show()