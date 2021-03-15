import gym
import numpy as np
from gym.utils import seeding
from termcolor import colored

from gym_rl_mpc.objects.pendulum import Pendulum

class PendulumEnv(gym.Env):
    """
    Creates an environment with a pendulum.
    """
    def __init__(self, env_config):
        print(colored('Debug: Initializing environment...', 'green'))
        for key in env_config:
            setattr(self, key, env_config[key])

        self.config = env_config

        self.action_space = gym.spaces.Box(low=np.array([-1, -0.2, 0]),
                                           high=np.array([1, 1, 1]),
                                           dtype=np.float32)

        # Legal limits for state observations
        low = np.array([    -np.pi,                             # theta
                            -np.finfo(np.float32).max,          # theta_dot
                            -np.finfo(np.float32).max,          # omega
                            0,                                  # wind speed
                        ])
        high = np.array([   np.pi,                              # theta
                            np.finfo(np.float32).max,           # theta_dot
                            np.finfo(np.float32).max,           # omega
                            self.max_wind_speed,                # wind speed
                        ])

        self.observation_space = gym.spaces.Box(low=low,
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
        self.pendulum.step(action, self.wind_speed)
        self.observation = self.observe()

        done, reward = self.calculate_reward(self.observation, action)

        self.cumulative_reward += reward
        self.last_reward = reward

        self.save_latest_step()

        self.t_step += 1

        return self.observation, reward, done, {}

    def generate_environment(self):
        """
        Generates environment with a pendulum at random initial conditions
        """
        self.wind_speed = 16#(self.max_wind_speed-self.min_wind_speed)*self.rand_num_gen.rand() + self.min_wind_speed
        self.pendulum = Pendulum(self.wind_speed, self.step_size)

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode is done.
        """
        done = False

        # theta_deg = self.pendulum.platform_angle*(180/np.pi)
        # theta_dot_deg_s = self.pendulum.state[1]*(180/np.pi)
        # omega_rpm = self.pendulum.state[2]*(60/(2*np.pi))
        # power_error_MegaWatts = np.abs(action[2]-self.pendulum.power_regime(self.wind_speed))*(self.pendulum.max_power_generation/1e6)

        # omega_ref_rpm = self.pendulum.omega_setpoint(self.wind_speed)*(60/(2*np.pi))
        # omega_error_rpm = np.abs(omega_rpm-omega_ref_rpm)

        self.theta_reward = 0 #np.exp(-self.gamma_theta*(np.abs(theta_deg))) - self.gamma_theta*theta_deg**2
        self.theta_dot_reward = 0 #-self.reward_theta_dot*theta_dot_deg_s**2
        self.omega_reward = 0 #np.exp(-self.gamma_omega*omega_error_rpm) - self.gamma_omega*omega_error_rpm**2
        self.power_reward = 0 #np.exp(-self.gamma_power*power_error_MegaWatts) - self.gamma_power*power_error_MegaWatts
        self.control_reward = 0 #-self.reward_control*(action[0]**2 + action[1]**2)

        theta = self.pendulum.platform_angle
        theta_dot = self.pendulum.state[1]
        omega = self.pendulum.state[2]
        power_error = np.abs(action[2]-self.pendulum.power_regime(self.wind_speed))

        omega_in_working_range = omega > self.pendulum.omega_setpoint(self.min_wind_speed) and omega < self.pendulum.omega_setpoint(self.max_wind_speed)
        theta_ok = np.abs(theta) < self.crash_angle_condition
        power_in_working_range = power_error < self.pendulum.max_power_generation

        survival_reward = self.reward_survival

        if omega_in_working_range and theta_ok and power_in_working_range:
            step_reward = survival_reward * self.working_range_reward_multiplier
        else:
            step_reward = survival_reward

        end_cond_2 = self.t_step >= self.max_episode_time/self.step_size
        crash_cond_1 = np.abs(self.pendulum.platform_angle) > self.crash_angle_condition
        crash_cond_2 = self.pendulum.omega > self.crash_omega_max
        crash_cond_3 = self.pendulum.omega < self.crash_omega_min

        if end_cond_2 or crash_cond_1 or crash_cond_2 or crash_cond_3:
            done = True
        if crash_cond_1 or crash_cond_2 or crash_cond_3:
            self.crashed = True

        return done, step_reward

    def observe(self):
        """Returns the array of observations at the current time-step.
        Returns
        -------
        obs : np.ndarray
            The observation of the environment.
        """
        obs = np.hstack([self.pendulum.state, self.wind_speed])
        return obs

    def seed(self, seed=None):
        """Reseeds the random number generator used in the environment"""
        self.rand_num_gen, seed = seeding.np_random(seed)
        return [seed]

    def save_latest_step(self):
        self.episode_history.setdefault('states', []).append(np.copy(self.pendulum.state[0:3]))
        self.episode_history.setdefault('states_dot', []).append(np.copy(self.pendulum.state[3:6]))
        self.episode_history.setdefault('input', []).append(self.pendulum.input)
        self.episode_history.setdefault('observations', []).append(self.observation)
        self.episode_history.setdefault('time', []).append(self.t_step*self.step_size)
        self.episode_history.setdefault('last_reward', []).append(self.last_reward)
        self.episode_history.setdefault('wind_force',[]).append(self.pendulum.wind_force)
        self.episode_history.setdefault('wind_torque',[]).append(self.pendulum.wind_torque)
        self.episode_history.setdefault('generator_torque',[]).append(self.pendulum.generator_torque)
        self.episode_history.setdefault('adjusted_wind_speed',[]).append(self.pendulum.adjusted_wind_speed)

        self.episode_history.setdefault('theta_reward',[]).append(self.theta_reward)
        self.episode_history.setdefault('theta_dot_reward',[]).append(self.theta_dot_reward)
        self.episode_history.setdefault('omega_reward',[]).append(self.omega_reward)
        self.episode_history.setdefault('power_reward',[]).append(self.power_reward)
        self.episode_history.setdefault('control_reward',[]).append(self.control_reward)

    def save_latest_episode(self):
        self.history = {
            'episode_num': self.episode,
            'avg_abs_theta': np.abs(np.array(self.episode_history['states'])[:, 0]).mean(),
            'std_theta': np.array(self.episode_history['states'])[:, 0].std(),
            'crashed': int(self.crashed),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'duration': self.t_step*self.step_size,
            'wind_speed': self.wind_speed,
            'theta_reward': np.array(self.episode_history['theta_reward']).mean(),
            'theta_dot_reward': np.array(self.episode_history['theta_dot_reward']).mean(),
            'omega_reward': np.array(self.episode_history['omega_reward']).mean(),
            'power_reward': np.array(self.episode_history['power_reward']).mean(),
            'control_reward': np.array(self.episode_history['control_reward']).mean(),
        }

        self.total_history.append(self.history)
