import multiprocessing

from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from gym_rl_mpc.objects.symbolic_model import change_random, get_sys, get_terminal_sys
from gym_rl_mpc.utils.model_params import DEG2RAD, RPM2RAD
from PSF.utils import plotEllipsoid, get_terminal_set, max_ellipsoid


def plot_terminal():
    sys = get_sys()
    P, _, x_0, _ = get_terminal_set(sys, get_terminal_sys())
    plotEllipsoid(P, sys["Hx"], sys["hx"], x_0, savedir="plots", name="ellipsoid_terminal")

    P_fake = max_ellipsoid(sys["Hx"], sys["hx"], x_0)
    plotEllipsoid(P_fake, sys["Hx"], sys["hx"], x_0, savedir="plots", name="ellipsoid_fake_terminal")


def plot_steady_state(pool, n=5000):
    results = pool.starmap(change_random, [() for _ in range(n)])
    z0s = np.hstack(results)
    z0s = z0s[:, ~np.isnan(z0s).any(axis=0)]

    W = z0s[-1, :] * 3 / 2
    Theta = z0s[0, :] / DEG2RAD

    Omega = z0s[2, :] / RPM2RAD

    fig1 = plt.figure()
    axs = fig1.add_subplot(projection='3d')
    axs.scatter(Theta, Omega, W, c=W, s=4, alpha=1)
    axs.set_xlabel(r'$\theta$ [Deg]')
    axs.set_ylabel(r'$\Omega$ [RPM]')
    axs.set_zlabel(r'Wind [m/s]')
    azim = 45
    axs.view_init(elev=40, azim=azim)
    plt.savefig(f'plots/steady_state_{azim}.pdf', bbox_inches='tight')
    azim += 90
    axs.view_init(elev=40, azim=azim)
    plt.savefig(f'plots/steady_state_{azim}.pdf', bbox_inches='tight')
    plt.show()

    '''
    triang = mtri.Triangulation(Theta, Omega)

    dist_Theta = Theta[triang.triangles].max(axis=1) - Theta[triang.triangles].min(axis=1)
    mask = dist_Theta > 0.5
    dist_Omega = Omega[triang.triangles].max(axis=1) - Omega[triang.triangles].min(axis=1)
    mask = np.logical_or(mask, (dist_Omega > 0.5))
    triang.set_mask(mask)
    fig2 = plt.figure()
    axt = fig2.add_subplot(projection='3d')
    axt.plot_trisurf(triang, W, cmap='viridis')

    axt.set_xlabel(r'$\theta$ [Deg]')
    axt.set_ylabel(r'$\Omega$ [RPM]')
    axt.set_zlabel(r'Wind [m/s]')

    # ax.plot(z0s[0, :], z0s[2, :], 'r+', zdir='y', zs=0)
    # ax.plot(z0s[1, :], z0s[2, :], 'g+', zdir='x', zs=0)
    # ax.plot(z0s[0, :], z0s[1, :], 'k+', zdir='z', zs=0)
    '''


if __name__ == "__main__":
    plot_terminal()
    #with multiprocessing.Pool(7) as p:
    #    plot_steady_state(p)
