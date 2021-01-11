import gym
import numpy as np


class TurbineStab3D(gym.Env):
    def __init__(self, env_config):
        self.reset()