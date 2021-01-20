import numpy as np
from gym_turbine.objects import turbine
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    height = 10
    wind_dir = 0
    action = np.array([0, 0, 0, 0])
    turbine.step(action, wind_dir)
    x = [0, height*np.sin(turbine.pitch)*np.cos(turbine.roll)]
    y = [0, height*np.sin(turbine.roll)*np.cos(turbine.pitch)]
    z = [0, height*np.cos(turbine.pitch)]
    plt.cla()
    # ax_ani.set_aspect('equal', adjustable='datalim')
    ax_ani.set(xlim=(-5, 5), ylim=(-5, 5), zlim=(0, 11))
    ax_ani.set_xlabel('$X$')
    ax_ani.set_ylabel('$Y$')
    ax_ani.set_zlabel('$Z$')

    plt.plot(x, y, z, color='r', linewidth=5)
    plt.plot(x, y, [z[1], z[1]], color='k', linewidth=1)


if __name__ == "__main__":

    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot(111, projection='3d')


    step_size = 0.01
    turbine = turbine.Turbine(step_size)

    # plot_figs(turbine)

    ani = FuncAnimation(fig_ani, animate, frames=200, interval=20, blit=False)
    plt.tight_layout()
    plt.show()
