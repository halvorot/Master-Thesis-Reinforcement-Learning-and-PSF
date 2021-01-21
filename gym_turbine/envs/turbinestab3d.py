import gym
import numpy as np
from gym.utils import seeding
from termcolor import colored

from gym_turbine.objects.turbine import Turbine

class TurbineStab3D(gym.Env):
    """
    Creates an environment with a turbine.
    """
    def __init__(self, env_config):
        print(colored('Debug: Initializing environment...', 'green'))
        for key in env_config:
            setattr(self, key, env_config[key])

        self.action_space = gym.spaces.Box(low=np.array([-1]*self.n_actuators),
                                           high=np.array([1]*self.n_actuators),
                                           dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=np.array([-1]*self.n_observations),
                                                high=np.array([1]*self.n_observations),
                                                dtype=np.float32)


        self.episode = 0
        self.total_t_steps = 0
        self.t_step = 0
        self.cumulative_reward = 0

        self.crashed = None
        self.last_reward = None

        self.rand_num_gen = None
        self.seed()

        self.reset()

    def reset(self):
        """
        Resets environment to initial state.
        """
        # Seeding
        if self.rand_num_gen is None:
            self.seed()

        # Incrementing counters
        self.episode += 1
        self.total_t_steps += self.t_step

        # Reset internal variables
        self.cumulative_reward = 0
        self.last_reward = 0
        self.t_step = 0
        self.crashed = False


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
        info = {}
        info['crashed'] = self.crashed
        self.cumulative_reward += reward
        self.last_reward = reward

        self.t_step += 1

        return self.observation, reward, done, info

    def generate_environment(self):
        """
        Generates environment with a turbine, and a random initial condition
        """
        init_roll = (2*self.rand_num_gen.rand()-1)*self.max_init_angle
        init_pitch = (2*self.rand_num_gen.rand()-1)*self.max_init_angle
        init_state = np.array([init_roll, init_pitch])
        self.turbine = Turbine(init_state, self.step_size)

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end.
        """
        done = False

        r_p = 1/(self.sigma_p*np.sqrt(2*np.pi)) * np.exp(-self.turbine.pitch**2 / (2*self.sigma_p**2))
        r_r = 1/(self.sigma_r*np.sqrt(2*np.pi)) * np.exp(-self.turbine.roll**2 / (2*self.sigma_r**2))
        reward_stab = r_p * r_r

        reward_power_use = 0    # -(action/self.max_input)**2

        step_reward = self.lambda_reward*reward_stab + (1-self.lambda_reward)*reward_power_use

        end_cond_1 = self.last_reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        crash_cond_1 = self.turbine.pitch > self.crash_angle_condition
        crash_cond_2 = self.turbine.roll > self.crash_angle_condition

        if end_cond_1 or end_cond_2 or crash_cond_1 or crash_cond_2:
            done = True
        if crash_cond_1 or crash_cond_2:
            step_reward = -1000

        return done, step_reward

    def observe(self):
        """Returns the array of observations at the current time-step.
        Returns
        -------
        obs : np.ndarray
            The observation of the environment.
        """
        roll = np.clip(self.turbine.roll / np.pi, -1, 1)
        pitch = np.clip(self.turbine.pitch / np.pi, -1, 1)
        obs = np.array([roll, pitch])
        return obs

    def seed(self, seed=None):
        """Reseeds the random number generator used in the environment"""
        self.rand_num_gen, seed = seeding.np_random(seed)
        return [seed]
