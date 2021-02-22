import gym
import numpy as np
from gym.utils import seeding
from termcolor import colored
import matplotlib.pyplot as plt

from gym_turbine.objects.turbine import Turbine

class TurbineEnv(gym.Env):
    """
    Creates an environment with a turbine.
    """
    def __init__(self, env_config):
        print(colored('Debug: Initializing environment...', 'green'))
        for key in env_config:
            setattr(self, key, env_config[key])

        self.config = env_config
        self.n_actuators = 4

        self.action_space = gym.spaces.Box(low=np.array([-1]*self.n_actuators),
                                           high=np.array([1]*self.n_actuators),
                                           dtype=np.float32)

        # Legal limits for state observations
        high = np.array([   np.finfo(np.float32).max,           # x_sg
                            np.finfo(np.float32).max,           # x_sw
                            np.finfo(np.float32).max,           # x_hv
                            np.pi,                              # theta_r
                            np.pi,                              # theta_p
                            np.finfo(np.float32).max,           # x_1
                            np.finfo(np.float32).max,           # x_2
                            np.finfo(np.float32).max,           # x_3
                            np.finfo(np.float32).max,           # x_4
                            np.finfo(np.float32).max,           # x_sg_dot
                            np.finfo(np.float32).max,           # x_sw_dot
                            np.finfo(np.float32).max,           # x_hv_dot
                            np.finfo(np.float32).max,           # theta_r_dot
                            np.finfo(np.float32).max,           # theta_p_dot
                            np.finfo(np.float32).max,           # x_1_dot
                            np.finfo(np.float32).max,           # x_2_dot
                            np.finfo(np.float32).max,           # x_3_dot
                            np.finfo(np.float32).max,           # x_4_dot
                        ])

        self.observation_space = gym.spaces.Box(low=-high,
                                                high=high,
                                                dtype=np.float32)


        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.cumulative_reward = 0

        self.total_history = []
        self.history = {}

        self.crashed = None
        self.last_reward = None

        self.rand_num_gen = None
        self.seed()

    def reset(self):
        """
        Resets environment to initial state.
        """
        # Seeding
        if self.rand_num_gen is None:
            self.seed()

        # Saving information about episode
        if self.t_step:
            self.save_latest_episode()

        # Incrementing counters
        self.episode += 1
        self.total_t_steps += self.t_step

        # Reset internal variables
        self.cumulative_reward = 0
        self.last_reward = 0
        self.t_step = 0
        self.crashed = False

        self.episode_history = {}

        self.generate_environment()
        self.observation = self.observe()

        return self.observation


    def step(self, action):
        """
        Simulates the environment one time-step.
        """

        self.turbine.step(action, self.wind_dir)
        self.observation = self.observe()

        done, reward = self.calculate_reward(self.observation, action)

        self.cumulative_reward += reward
        self.last_reward = reward

        self.save_latest_step()

        self.t_step += 1

        return self.observation, reward, done, {}

    def generate_environment(self):
        """
        Generates environment with a turbine at random initial conditions
        """
        init_roll = (2*self.rand_num_gen.rand()-1)*self.max_init_angle      # random number in range (+- max_init_angle)
        init_pitch = (2*self.rand_num_gen.rand()-1)*self.max_init_angle     # random number in range (+- max_init_angle)
        init_state = np.array([init_roll, init_pitch])
        self.turbine = Turbine(init_state, self.step_size)

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode is done.
        """
        done = False

        x = self.turbine.pitch
        y = self.turbine.roll
        # r_p = np.exp(-self.gamma_p*x**2)
        # r_r = np.exp(-self.gamma_r*y**2)

        # step_reward = np.minimum(1, -np.log(self.gamma_p*np.abs(x)+1/self.gamma_p)) + np.minimum(1, -np.log(self.gamma_r*np.abs(y)+1/self.gamma_r))

        r_p = np.exp(-self.gamma_p*(np.abs(x)))
        r_r = np.exp(-self.gamma_r*(np.abs(y)))
        step_reward = r_p * r_r - 2*(self.gamma_p*x**2 + self.gamma_r*y**2)

        end_cond_1 = self.cumulative_reward < self.min_reward
        end_cond_2 = self.t_step >= self.max_t_steps
        crash_cond_1 = np.abs(self.turbine.pitch) > self.crash_angle_condition
        crash_cond_2 = np.abs(self.turbine.roll) > self.crash_angle_condition

        if end_cond_1 or end_cond_2 or crash_cond_1 or crash_cond_2:
            done = True
        if crash_cond_1 or crash_cond_2:
            step_reward = self.reward_crash
            self.crashed = True

        return done, step_reward

    def observe(self):
        """Returns the array of observations at the current time-step.
        Returns
        -------
        obs : np.ndarray
            The observation of the environment.
            Includes x and x_dot, where x = [x_sg, x_sw, x_hv, theta_r, theta_p]
        """
        # roll = np.clip(self.turbine.roll / np.pi, -1, 1)        # Clip at +-180 degrees
        # pitch = np.clip(self.turbine.pitch / np.pi, -1, 1)      # Clip at +-180 degrees

        obs = self.turbine.state
        return obs

    def seed(self, seed=None):
        """Reseeds the random number generator used in the environment"""
        self.rand_num_gen, seed = seeding.np_random(seed)
        return [seed]

    def render(self):

        x_surface = self.turbine.position[0]
        y_surface = self.turbine.position[1]
        z_surface = self.turbine.position[2]
        x_top = x_surface + self.turbine.height*np.sin(self.turbine.pitch)*np.cos(self.turbine.roll)
        y_top = -(y_surface + self.turbine.height*np.sin(self.turbine.roll)*np.cos(self.turbine.pitch))
        z_top = z_surface + self.turbine.height*np.cos(self.turbine.pitch)

        x = [x_surface, x_top]
        y = [y_surface, y_top]
        z = [z_surface, z_top]
        x_base = [-0.2*(x_top-x_surface) + x_surface, x_surface]
        y_base = [-0.2*(y_top-y_surface) + y_surface, y_surface]
        z_base = [-0.2*(z_top-z_surface) + z_surface, z_surface]

        ax = plt.axes(projection='3d')
        ax.plot(x, y, z)
        ax.plot(x_base, y_base, z_base, color='r', linewidth=10)
        return ax

    def save_latest_step(self):
        self.episode_history.setdefault('states', []).append(np.copy(self.turbine.state[0:9]))
        self.episode_history.setdefault('states_dot', []).append(np.copy(self.turbine.state[9:18]))
        self.episode_history.setdefault('input', []).append(self.turbine.input)
        self.episode_history.setdefault('observations', []).append(self.observation)
        self.episode_history.setdefault('time', []).append(self.t_step*self.step_size)
        self.episode_history.setdefault('last_reward', []).append(self.last_reward)

    def save_latest_episode(self):
        self.history = {
            'episode_num': self.episode,
            'avg_abs_theta_r': np.abs(np.array(self.episode_history['states'])[:, 3]).mean(),
            'avg_abs_theta_p': np.abs(np.array(self.episode_history['states'])[:, 4]).mean(),
            'crashed': int(self.crashed),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'duration': self.t_step*self.step_size
        }

        self.total_history.append(self.history)
