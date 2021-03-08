import numpy as np
import gym
from stable_baselines3 import PPO
from gym_rl_mpc.utils import model_params as params
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib.animation import FuncAnimation
import pandas as pd
import argparse
import os

def animate(frame):
    plt.cla()
    height = params.L

    if args.data:
        # If reading from file:
        x_top = height*np.sin(data_angle[frame])
        y_top = -(height*np.cos(data_angle[frame]))
        action = data_input[0][frame]/params.max_thrust_force
    else:
        if args.agent:
            action, _states = agent.predict(env.observation, deterministic=True)
        else:
            action = np.array([0,0])
            
        _, _, done, _ = env.step(action)
        if done:
            print("Environment done")
            raise SystemExit
        x_top = height*np.sin(env.pendulum.platform_angle)
        y_top = height*np.cos(env.pendulum.platform_angle)
        x_bottom = -params.L_thr*np.sin(env.pendulum.platform_angle)
        y_bottom = -params.L_thr*np.cos(env.pendulum.platform_angle)
        recorded_states.append(env.pendulum.state)
        recorded_inputs.append(env.pendulum.input)
        recorded_disturbance.append(np.array([env.pendulum.wind_force, env.pendulum.wind_torque]))

    x = [x_bottom, x_top]
    y = [y_bottom, y_top]
    ax_ani.set(xlim=(-0.7*height, 0.7*height), ylim=(-1.1*params.L_thr, 1.1*height))
    ax_ani.set_xlabel('$X$')
    ax_ani.set_ylabel('$Y$')

    # Plot water line
    ax_ani.plot([-0.6*height, 0.6*height], [0, 0], linewidth=1, linestyle='--')

    # Plot pole
    ax_ani.plot(x, y, color='b', linewidth=2)
    # Plot arrow proportional to input force
    ax_ani.arrow(x = -params.L_thr*np.sin(env.pendulum.platform_angle), y = -params.L_thr*np.cos(env.pendulum.platform_angle), dx=100*action[0], dy=0, head_width=2, head_length=2, length_includes_head=True)
    # Plot arrow proportional to wind force
    ax_ani.arrow(x = x_top, y = y_top, dx=100*(env.pendulum.wind_force/params.max_wind_force), dy=0, head_width=2, head_length=2, length_includes_head=True)
    # Plot wind arrow with wind number
    ax_ani.arrow(x = -50, y = params.L, dx=20, dy=0, head_width=2, head_length=2, length_includes_head=True)
    ax_ani.arrow(x = -50, y = params.L-10, dx=20, dy=0, head_width=2, head_length=2, length_includes_head=True)
    ax_ani.text(-49, params.L-7, f"{env.wind_speed:.1f} m/s", fontsize=10)
    # Plot rotational speed
    ax_ani.text(0, params.L+7, f"$\Omega$ = {env.pendulum.omega*(180/np.pi):.1f} deg/s", fontsize=10)


if __name__ == "__main__":
    fig_ani = plt.figure()
    ax_ani = fig_ani.add_subplot(111)

    recorded_states = []
    recorded_inputs = []
    recorded_disturbance = []

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
        data_angle = data['theta']
        data_input = np.array([data['F']])
        data_reward = np.array(data['reward'])
        env_id = "PendulumStab-v1"
    else:
        done = False
        env = gym.make("PendulumStab-v1")
        env_id = env.unwrapped.spec.id
        env.reset()
        if args.agent:
            agent = PPO.load(args.agent)

    animation_speed = 10
    ani = FuncAnimation(fig_ani, animate, interval=1000*env.step_size/animation_speed, blit=False)

    plt.tight_layout()
    if args.save_video:
        agent_path_list = args.agent.split("\\")
        video_dir = os.path.join("logs", env_id, agent_path_list[-3], "videos")
        os.makedirs(video_dir, exist_ok=True)
        i = 0
        video_path = os.path.join(video_dir, agent_path_list[-1][0:-4] + f"_animation_{i}.mp4")
        while os.path.exists(video_path):
            i += 1
            video_path = os.path.join(video_dir, agent_path_list[-1][0:-4] + f"_animation_{i}.mp4")
        ani.save(video_path, dpi=150)
    plt.show()

    if not args.data:
        recorded_states = np.array(recorded_states)
        recorded_inputs = np.array(recorded_inputs)
        recorded_disturbance = np.array(recorded_disturbance)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle(f"Wind speed: {env.wind_speed:.1f} m/s")

        time = np.array(range(0, len(recorded_states[:,0])))*env.step_size

        ax1.plot(time, recorded_states[:,0]*(180/np.pi), label='theta')
        ax1.plot(time, recorded_states[:,1]*(180/np.pi), label='theta_dot')
        ax1.plot(time, np.zeros(len(time)), linestyle='--', color='k')
        ax1.set_ylabel('Degrees, deg/sec')
        ax1.set_title('platform angle and angular velocity')
        ax1.legend()

        ax2.plot(time, recorded_states[:,2]*(180/np.pi), label='omega')
        ax2.set_ylabel('Degrees/sec')
        ax2.set_title('Angluar Velocity Rotor')
        ax2.legend()

        color = 'tab:blue'
        ax3.plot(time, recorded_inputs[:,0], label='F_thr', color=color)
        ax3.set_ylabel('F_thr [N]', color=color)
        ax3.set_title('Input')
        ax3.legend()

        color = 'tab:orange'
        ax3_2 = ax3.twinx()
        ax3_2.plot(time, recorded_inputs[:,1]*(180/np.pi), label='Blade pitch', color=color)
        ax3_2.set_ylabel('Blade pitch [Degrees]', color=color)
        ax3_2.legend()

        color = 'tab:blue'
        ax4.plot(time, recorded_disturbance[:,0], label='F_w', color=color)
        ax4.plot(time, np.zeros(len(time)), linestyle='--', color='k', linewidth=0.5)
        ax4.set_ylabel('F_w [N]', color=color)
        ax4.set_title('Wind')
        ax4.legend()

        color = 'tab:orange'
        ax4_2 = ax4.twinx()
        ax4_2.plot(time, recorded_disturbance[:,1], label='Q_w', color=color)
        ax4_2.set_ylabel('Q_w [Nm]', color=color)
        ax4_2.legend()

        plt.show()
