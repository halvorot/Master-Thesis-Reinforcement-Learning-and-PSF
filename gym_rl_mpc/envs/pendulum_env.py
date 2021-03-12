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
        init_angle = (2*self.rand_num_gen.rand()-1)*self.max_init_angle     # random number in range (+- max_init_angle)
        self.wind_speed = (self.max_wind_speed-self.min_wind_speed)*self.rand_num_gen.rand() + self.min_wind_speed
        self.pendulum = Pendulum(init_angle, self.wind_speed, self.step_size)

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode is done.
        """
        done = False

        theta = self.pendulum.platform_angle
        theta_dot = self.pendulum.state[1]
        omega = self.pendulum.state[2]
        power = action[2]

        theta_reward = np.exp(-self.gamma*(np.abs(theta))) - self.gamma*theta**2
        theta_dot_reward = -self.reward_theta_dot*theta_dot**2
        omega_reward = -self.reward_omega*(omega-self.pendulum.omega_setpoint(self.wind_speed))**2
        power_reward = -self.reward_power*(power-self.pendulum.power_regime(self.wind_speed))**2

        control_reward = -self.reward_control*(action[0]**2 + action[1]**2)

        step_reward = theta_reward + theta_dot_reward + omega_reward + control_reward + power_reward

        end_cond_2 = self.t_step >= self.max_episode_time/self.step_size
        crash_cond_1 = np.abs(self.pendulum.platform_angle) > self.crash_angle_condition
        crash_cond_2 = self.pendulum.omega > self.crash_omega_max
        crash_cond_3 = self.pendulum.omega < self.crash_omega_min

        if end_cond_2 or crash_cond_1 or crash_cond_2 or crash_cond_3:
            done = True
        if crash_cond_1 or crash_cond_2 or crash_cond_3:
            step_reward = self.reward_crash
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

    def save_latest_episode(self):
        self.history = {
            'episode_num': self.episode,
            'avg_abs_theta': np.abs(np.array(self.episode_history['states'])[:, 0]).mean(),
            'std_theta': np.array(self.episode_history['states'])[:, 0].std(),
            'crashed': int(self.crashed),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'duration': self.t_step*self.step_size,
            'wind_speed': self.wind_speed
        }

        self.total_history.append(self.history)
