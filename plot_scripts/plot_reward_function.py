import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import argparse

RAD2DEG = 180/np.pi

def rew_func(x, y, gamma):
    return np.exp(-gamma*(x**2)) * np.exp(-gamma*(y**2))

def plot_r_stab_3d(gamma, save=False):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X = np.arange(-0.2, 0.2, 0.01)
    Y = np.arange(-0.2, 0.2, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = rew_func(X, Y, gamma)

    # Plot the surface.
    surf = ax.plot_surface(X*RAD2DEG, Y*RAD2DEG, Z, cmap='magma',
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0, 1.5)
    ax.zaxis.set_major_locator(LinearLocator(10))   # Number or ticks on Z-axis
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))  # Format of numbers on Z-axis

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Add axis labels
    ax.set_xlabel(r'$\theta_p$ [deg]')
    ax.set_ylabel(r'$\theta_r$ [deg]')
    ax.set_zlabel('Reward')
    ax.view_init(elev=30, azim=-71)
    if save:
        plt.savefig('plot_results/r_stab_alternative_plot_3d.pdf', bbox_inches='tight')

def plot_r_stab_contour(gamma, save=False):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make data.
    X = np.arange(-0.2, 0.2, 0.01)
    Y = np.arange(-0.2, 0.2, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = rew_func(X, Y, gamma)

    # Plot the surface.
    reward_plot = ax.contourf(X*RAD2DEG, Y*RAD2DEG, Z, levels=20, cmap='magma')

    cbar = fig.colorbar(reward_plot)
    cbar.set_label(r'Reward')

    # Add axis labels
    ax.set_xlabel(r'$\theta_p$ [deg]')
    ax.set_ylabel(r'$\theta_r$ [deg]')
    if save:
        plt.savefig('plot_results/r_stab_plot_alternative_contour.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save',
        help='Save plots',
        action='store_true'
    )
    args = parser.parse_args()

    gamma = 250
    plot_r_stab_3d(gamma, save=args.save)
    plot_r_stab_contour(gamma, save=args.save)
    plt.show()
