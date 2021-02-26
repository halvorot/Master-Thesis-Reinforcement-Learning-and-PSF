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
        self.n_actuators = 2

        self.action_space = gym.spaces.Box(low=np.array([-1]*self.n_actuators),
                                           high=np.array([1]*self.n_actuators),
                                           dtype=np.float32)

        # Legal limits for state observations
        high = np.array([   np.pi,                              # theta
                            np.finfo(np.float32).max,           # x_1
                            np.finfo(np.float32).max,           # x_2
                            np.finfo(np.float32).max,           # theta_dot
                            np.finfo(np.float32).max,           # x_1_dot
                            np.finfo(np.float32).max,           # x_2_dot
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

        self.pendulum.step(action)
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
        init_pitch = (2*self.rand_num_gen.rand()-1)*self.max_init_angle     # random number in range (+- max_init_angle)
        self.pendulum = Pendulum(init_pitch, self.step_size)

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode is done.
        """
        done = False

        x = self.pendulum.pitch

        step_reward = np.exp(-self.gamma*(np.abs(x)))

        end_cond_1 = self.cumulative_reward < self.min_reward
        end_cond_2 = self.t_step >= self.max_t_steps
        crash_cond = np.abs(self.pendulum.pitch) > self.crash_angle_condition

        if end_cond_1 or end_cond_2 or crash_cond:
            done = True
        if crash_cond:
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
        obs = self.pendulum.state
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

    def save_latest_episode(self):
        self.history = {
            'episode_num': self.episode,
            'avg_abs_theta': np.abs(np.array(self.episode_history['states'])[:, 0]).mean(),
            'crashed': int(self.crashed),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'duration': self.t_step*self.step_size
        }

        self.total_history.append(self.history)
