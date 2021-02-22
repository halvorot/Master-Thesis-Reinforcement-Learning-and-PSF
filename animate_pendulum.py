import numpy as np
import gym
from stable_baselines3 import PPO
from gym_turbine.utils import state_space_pendulum as ss
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.animation import FuncAnimation
import pandas as pd
import argparse
import os

def plot_states(pendulum, sim_time):
    step_size = 0.01
    N = int(sim_time/step_size)
    pitch = np.array([])
    dva_input = []

    for i in range(N):
        action = np.array([0, 0])
        pendulum.step(action)
        pitch = np.append(pitch, pendulum.pitch)
        dva_input.append(pendulum.input)

    dva_input = np.asarray(dva_input)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    time = np.linspace(0, N*step_size, pitch.size)
    ax1.plot(time, pitch*(180/np.pi), label='Pitch')
    ax2.plot(time, dva_input[:, 0], label='DVA#1')
    ax2.plot(time, dva_input[:, 1], label='DVA#2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def animate(frame):
    plt.cla()
    height = ss.height
    spoke_length = ss.spoke_length

    if args.data:
        # If reading from file:
        x_top = height*np.sin(data_pitch[frame])
        y_top = -(height*np.cos(data_pitch[frame]))
        action = np.array([data_input[0][frame]/ss.max_input, data_input[1][frame]/ss.max_input])
        dva_displacement = data_dva_displacement
    else:
        if args.agent:
            action, _states = agent.predict(env.observation, deterministic=True)
        else:
            if frame > 50:
                action = np.array([1, 0])
            else:
                action = np.array([0, 0])
            action = np.array([0, 0])

        _, _, done, _ = env.step(action)
        if done:
            print("Environment done")
            raise SystemExit
        x_top = height*np.sin(env.pendulum.pitch)
        y_top = height*np.cos(env.pendulum.pitch)
        recorded_states.append(env.pendulum.state[0:3])
        recorded_inputs.append(env.pendulum.input)
        dva_displacement = env.pendulum.dva_displacement

    x = [0, x_top]
    y = [0, y_top]
    ax_ani.set(xlim=(-0.7*height, 0.7*height), ylim=(-0.2*height, 1.1*height))
    ax_ani.set_xlabel('$X$')
    ax_ani.set_ylabel('$Y$')

    # Plot water line
    ax_ani.plot([-0.6*height, 0.6*height], [0, 0], linewidth=1, linestyle='--')

    # Plot pole
    ax_ani.plot(x, y, color='b', linewidth=2)
    # Plot base
    base_thickness = 5
    base = Rectangle((-ss.spoke_length, -base_thickness/2), 2*ss.spoke_length, base_thickness)
    base.set_transform(mpl.transforms.Affine2D().rotate_deg_around(0, 0, -env.pendulum.pitch*180/np.pi) + ax_ani.transData)
    ax_ani.add_patch(base)
    # Plot line from neutral top position to current top position
    ax_ani.plot([0, x_top], [0, y_top], color='k', linewidth=1)

    # Plot arrow proportional to DVA displacement
    ax_ani.arrow(x = -spoke_length, y = 0, dx=0, dy=2*dva_displacement[0])
    ax_ani.arrow(x = spoke_length, y = 0, dx=0, dy=2*dva_displacement[1])


if __name__ == "__main__":
    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot(111)

    state_labels = np.array([r"theta", r"x_1", r"x_2"])
    input_labels = [r"Fa_1", r"Fa_2"]
    recorded_states = []
    recorded_inputs = []

    parser = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group()
    parser_group.add_argument(
        '--data',
        help='Path to data .csv file.',
    )
    parser_group.add_argument(
        '--agent',
        help='Path to agent .zip file.',
    )
    parser.add_argument(
        '--time',
        type=int,
        default=50,
        help='Max simulation time (seconds).',
    )
    parser.add_argument(
        '--save_video',
        help='Save animation as mp4 file',
        action='store_true'
    )
    args = parser.parse_args()

    if args.data:
        # If file specified, read data from file and animate
        data = pd.read_csv(args.data)
        data_pitch = data['theta']
        data_input = np.array([data['Fa_1'], data['Fa_2']])
        data_dva_displacement = np.array([data['x_1'], data['x_2']])
        data_reward = np.array(data['reward'])
    else:
        done = False
        env = gym.make("PendulumStab-v0")
        env.reset()
        if args.agent:
            agent = PPO.load(args.agent)
        else:
            pass
            # env.pendulum.state = np.zeros(6)
        recorded_states.append(env.pendulum.state[0:3])
        recorded_inputs.append(env.pendulum.input)

    ani = FuncAnimation(fig_ani, animate, interval=50, blit=False)

    plt.tight_layout()
    if args.save_video:
        agent_path_list = args.agent.split("\\")
        video_dir = os.path.join("logs", "PendulumStab-v0", agent_path_list[-3], "videos")
        os.makedirs(video_dir, exist_ok=True)
        i = 0
        video_path = os.path.join(video_dir, agent_path_list[-1][0:-4] + f"_animation_{i}.mp4")
        while os.path.exists(video_path):
            i += 1
            video_path = os.path.join(video_dir, agent_path_list[-1][0:-4] + f"_animation_{i}.mp4")
        ani.save(video_path, dpi=150)
    plt.show()

    if not args.data:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        rec_data = pd.DataFrame(np.hstack([recorded_states, recorded_inputs]), columns=np.hstack([state_labels, input_labels]))

        ax1.plot(rec_data['theta']*(180/np.pi), label='Pitch')
        ax1.set_ylabel('Degrees')
        ax1.set_title('Angles')
        ax1.legend()

        ax3.plot(rec_data['Fa_1'], label='Fa_1', linestyle='--')
        ax3.plot(rec_data['Fa_2'], label='Fa_2')
        ax3.set_ylabel('[N]')
        ax3.set_title('Inputs')
        ax3.legend()

        ax4.plot(rec_data['x_1'], label='x_1', linestyle='--')
        ax4.plot(rec_data['x_2'], label='x_2')
        ax4.set_ylabel('Meters')
        ax4.set_title('DVA displacements')
        ax4.legend()

        plt.show()
