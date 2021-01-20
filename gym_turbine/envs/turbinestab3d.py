import gym
import numpy as np


class TurbineStab3D(gym.Env):
    def __init__(self, env_config):
        self.n_observations = 2     # theta_pitch and theta_roll

        self.action_space = gym.spaces.Box(low=np.array([-1]*self.n_actuators),
                                           high=np.array([1]*self.n_actuators),
                                           dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=np.array([-1]*self.n_observations),
                                                high=np.array([1]*self.n_observations),
                                                dtype=np.float32)

        self.reset()

    def reset(self):
        self.cumulative_reward = 0
        self.t_step = 0
        self.last_reward = 0
        self.crash = False

        self.observation = None

    def step(self, action):
        """
        Simulates the environment one time-step.
        """

        # Simulate Wind
        self.wind.sim()
        wind_dir = self.current(self.turbine.state)
        self.current_history.append(wind_dir)

        self.turbine.step(action, wind_dir)

        done, reward = self.calculate_reward(self.observation, action)
        info = {}
        info['crash'] = self.crash
        self.cumulative_reward += reward

        self.t_step += 1

        return self.observation, reward, done, info

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end.
        """
        done = False

        r_p = 1/(self.sigma_p*np.sqrt(2*np.pi)) * np.exp(-self.turbine.pitch**2 / (2*self.sigma_p**2))
        r_r = 1/(self.sigma_r*np.sqrt(2*np.pi)) * np.exp(-self.turbine.roll**2 / (2*self.sigma_r**2))
        reward_stab = r_p * r_r

        reward_power_use = -(self.action**2)

        step_reward = self.lambda_reward*reward_stab + (1-self.lambda_reward)*reward_power_use

        end_cond_1 = self.reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = self.turbine.pitch > self.crash_angle_condition
        end_cond_4 = self.turbine.roll > self.crash_angle_condition

        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            done = True

        return done, step_reward
