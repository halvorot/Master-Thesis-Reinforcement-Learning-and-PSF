import numpy as np
from gym_turbine.objects import turbine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

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
        dva_input.append(turbine.dva_input)

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
    height = 10
    wind_dir = 0
    if frame in [30,31,32,33,34,35,36,37,38,39,40]:
        action = np.array([0, 0, 0, 0])
    else:
        action = np.array([0, 0, 0, 0])

    turbine.step(action, wind_dir)
    x_val = height*np.sin(turbine.pitch)*np.cos(turbine.roll)
    y_val = -height*np.sin(turbine.roll)*np.cos(turbine.pitch)
    z_val = height*np.cos(turbine.pitch)
    x = [0, x_val]
    y = [0, y_val]
    z = [0, z_val]
    x_base = [-0.15*x_val, 0]
    y_base = [-0.15*y_val, 0]
    z_base = [-0.15*z_val, 0]
    plt.cla()
    # ax_ani.set_aspect('equal', adjustable='datalim')
    ax_ani.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(0, 11))
    ax_ani.set_xlabel('$X$')
    ax_ani.set_ylabel('$Y$')
    ax_ani.set_zlabel('$Z$')

    plt.plot(x, y, z, color='b', linewidth=5)
    plt.plot(x, y, [z[1], z[1]], color='k', linewidth=1)
    plt.plot(x_base, y_base, z_base, color='r', linewidth=10)
    plt.tight_layout()
    # print('Sim time', time.time() - start_time)


if __name__ == "__main__":

    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot(111, projection='3d')


    step_size = 0.01
    turbine = turbine.Turbine(step_size)

    # plot_figs(turbine)

    ani = FuncAnimation(fig_ani, animate, interval=(1/30)*1000, blit=False)
    plt.tight_layout()
    plt.show()
