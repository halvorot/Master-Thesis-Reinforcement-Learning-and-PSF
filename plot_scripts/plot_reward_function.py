import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plot_r_stab_3d(gamma, sigma_p, sigma_r, save=False):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    X = np.arange(-0.5, 0.5, 0.01)
    Y = np.arange(-0.5, 0.5, 0.01)
    X, Y = np.meshgrid(X, Y)
    r_p = np.exp(-gamma*(X**2))  # 1/(sigma_p*np.sqrt(2*np.pi)) * np.exp(-X**2 / (2*sigma_p**2))
    r_r = np.exp(-gamma*(Y**2))  # 1/(sigma_r*np.sqrt(2*np.pi)) * np.exp(-Y**2 / (2*sigma_r**2))
    Z = r_p * r_r

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap='magma',
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0, 1.5)
    ax.zaxis.set_major_locator(LinearLocator(10))   # Number or ticks on Z-axis
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # Add axis labels
    ax.set_xlabel(r'$\theta_p$ [rad]')
    ax.set_ylabel(r'$\theta_r$ [rad]')
    ax.set_zlabel('Reward')
    ax.view_init(elev=30, azim=-71)
    if save:
        plt.savefig('../plot_results/r_stab_alternative_plot_3d.pdf', bbox_inches='tight')

def plot_r_stab_contour(gamma, sigma_p, sigma_r, save=False):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Make data.
    X = np.arange(-0.5, 0.5, 0.01)
    Y = np.arange(-0.5, 0.5, 0.01)
    X, Y = np.meshgrid(X, Y)
    r_p = np.exp(-gamma*(X**2))  # 1/(sigma_p*np.sqrt(2*np.pi)) * np.exp(-X**2 / (2*sigma_p**2))
    r_r = np.exp(-gamma*(Y**2))  # 1/(sigma_r*np.sqrt(2*np.pi)) * np.exp(-Y**2 / (2*sigma_r**2))
    Z = r_p * r_r

    # Plot the surface.
    reward_plot = ax.contourf(X, Y, Z, levels=20, cmap='magma')

    cbar = fig.colorbar(reward_plot)
    cbar.set_label(r'Reward')

    # Add axis labels
    ax.set_xlabel(r'$\theta_p$ [rad]')
    ax.set_ylabel(r'$\theta_r$ [rad]')
    if save:
        plt.savefig('../plot_results/r_stab_plot_alternative_contour.pdf', bbox_inches='tight')

sigma_p = 0.2
sigma_r = 0.2
gamma = 15
plot_r_stab_3d(gamma, sigma_p, sigma_r, save=True)
plot_r_stab_contour(gamma, sigma_p, sigma_r, save=True)
plt.show()
