import gym
import numpy as np
from gym.utils import seeding
from termcolor import colored

from gym_rl_mpc.envs.base_turbine_env import BaseTurbineEnv
from gym_rl_mpc.objects.turbine import Turbine
import gym_rl_mpc.utils.model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM, RPM2RAD, DEG2RAD
import gym_rl_mpc.objects.symbolic_model as sym
from PSF.PSF import PSF


class ConstantWind(BaseTurbineEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def generate_environment(self):
        """
        Generates environment with a turbine and a random wind speed between min and max wind speed in config
        """
        self.wind_speed = (self.max_wind_speed - self.min_wind_speed) * self.rand_num_gen.rand() + self.min_wind_speed
        self.turbine = Turbine(self.wind_speed, self.step_size)

class VariableWind(BaseTurbineEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wind_amplitude = min((self.max_wind_speed-self.min_wind_speed)/2, self.max_wind_amplitude) * self.rand_num_gen.rand()
        self.wind_mean = (self.max_wind_speed - self.min_wind_speed - 2*self.wind_amplitude) * self.rand_num_gen.rand() + self.min_wind_speed + self.wind_amplitude
        self.wind_phase_shift = 2*np.pi*self.rand_num_gen.rand()

    def step(self, action):
        t = self.t_step*self.step_size

        wind_speed = self.wind_amplitude*np.sin((1/self.wind_period)*2*np.pi*t + self.wind_phase_shift) + self.wind_mean + np.random.normal(0, 0.1*self.wind_amplitude)
        self.wind_speed = np.clip(wind_speed, self.min_wind_speed, self.max_wind_speed)

        return super().step(action)


    def generate_environment(self):
        """
        Generates environment with a turbine and a random wind speed between min and max wind speed in config
        """
        self.wind_speed = (self.max_wind_speed - self.min_wind_speed) * self.rand_num_gen.rand() + self.min_wind_speed
        self.turbine = Turbine(self.wind_speed, self.step_size)