import numpy as np
from gym_turbine.objects import turbine
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
import pandas as pd
import argparse

def plot_figs(turbine):
    sim_time = 60
    step_size = 0.01
    N = int(sim_time/step_size)
    pitch = np.array([])
    roll = np.array([])
    dva_input = []

    for i in range(N):
        wind_dir = 0
        action = np.array([0.9*np.sin(0.001*i), 0, -np.sin(0.001*i), 0])
        turbine.step(action, wind_dir)
        pitch = np.append(pitch, turbine.pitch)
        roll = np.append(roll, turbine.roll)
        dva_input.append(turbine.input)

    dva_input = np.asarray(dva_input)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    time = np.linspace(0, N*step_size, pitch.size)
    ax1.plot(time, pitch*(180/np.pi), label='Pitch')
    ax1.plot(time, roll*(180/np.pi), label='Pitch')
    ax2.plot(time, dva_input[:, 0], label='DVA#1')
    ax2.plot(time, dva_input[:, 1], label='DVA#2')
    ax2.plot(time, dva_input[:, 2], label='DVA#3')
    ax2.plot(time, dva_input[:, 3], label='DVA#4')
    plt.legend()
    plt.tight_layout()
    plt.show()

def animate(frame):
    start_time = time.time()
    height = 90
    wind_dir = 0

    if args.data:
        # If reading from file:
        x_surface = data_position[0][frame]
        y_surface = data_position[1][frame]
        z_surface = data_position[2][frame]
        x_top = x_surface + height*np.sin(data_pitch[frame])*np.cos(data_roll[frame])
        y_top = -(y_surface + height*np.sin(data_roll[frame])*np.cos(data_pitch[frame]))
        z_top = z_surface + height*np.cos(data_pitch[frame])
    else:
        ## Simulate turbine step by step ##
        action = np.array([0, 0, 0, 0])
        turbine.step(action, wind_dir)
        x_surface = turbine.position[0]
        y_surface = turbine.position[1]
        z_surface = turbine.position[2]
        x_top = x_surface + height*np.sin(turbine.pitch)*np.cos(turbine.roll)
        y_top = -(y_surface + height*np.sin(turbine.roll)*np.cos(turbine.pitch))
        z_top = z_surface + height*np.cos(turbine.pitch)

    x = [x_surface, x_top]
    y = [y_surface, y_top]
    z = [z_surface, z_top]
    x_base = [-0.2*(x_top-x_surface) + x_surface, x_surface]
    y_base = [-0.2*(y_top-y_surface) + y_surface, y_surface]
    z_base = [-0.2*(z_top-z_surface) + z_surface, z_surface]
    plt.cla()
    # ax_ani.set_aspect('equal', adjustable='datalim')
    ax_ani.set(xlim=(-2*height, 2*height), ylim=(-2*height, 2*height), zlim=(-height, 2*height))
    ax_ani.set_xlabel('$X$')
    ax_ani.set_ylabel('$Y$')
    ax_ani.set_zlabel('$Z$')

    # Plot surface (water)
    surface_X = np.arange(-2*height, 2*height+1, height)
    surface_Y = np.arange(-2*height, 2*height+1, height)
    surface_X, surface_Y = np.meshgrid(surface_X, surface_Y)
    surface_Z = 0*surface_X
    ax_ani.plot_surface(surface_X, surface_Y, surface_Z, alpha=0.1, linewidth=0, antialiased=False)

    # Plot pole
    plt.plot(x, y, z, color='b', linewidth=5)
    # Plot base
    plt.plot(x_base, y_base, z_base, color='r', linewidth=10)
    # Plot line from neutral top position to current top position
    plt.plot([0, x_top], [0, y_top], [height, z_top], color='k', linewidth=1)
    # Plot line from neutral base position to current base position
    plt.plot([0, x_surface], [0, y_surface], [0, z_surface], color='k', linewidth=1)
    plt.tight_layout()
    # print('Sim time', time.time() - start_time)


if __name__ == "__main__":

    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot(111, projection='3d')
    ax_ani.view_init(elev=18, azim=45)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        help='Path to data .csv file.',
    )
    args = parser.parse_args()
    if args.data:
        # If file specified, read data from file and animate
        data = pd.read_csv(args.data)
        data_position = np.array([data['x_sg'], data['x_sw'], data['x_hv']])
        data_roll = data['theta_r']
        data_pitch = data['theta_p']
    else:
        # If not file specified, simulate turbine step by step and animate
        step_size = 0.01
        init_roll = 0   # 10*(np.pi/180)
        init_pitch = 10*(np.pi/180)
        turbine = turbine.Turbine(np.array([init_roll, init_pitch]), step_size)

    ani = FuncAnimation(fig_ani, animate, interval=10, blit=False)
    plt.tight_layout()
    plt.show()
