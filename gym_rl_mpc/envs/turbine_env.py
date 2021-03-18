from pathlib import Path

import gym
import numpy as np
from gym.utils import seeding
from termcolor import colored

from gym_rl_mpc.objects.turbine import Turbine
import gym_rl_mpc.utils.model_params as params
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM, RPM2RAD, DEG2RAD
import gym_rl_mpc.objects.symbolic_model as sym
from casadi import SX, Function, jacobian
from PSF.PSF import PSF

class TurbineEnv(gym.Env):
    """
    Creates an environment with a turbine.
    """

    def __init__(self, env_config):
        print(colored('Debug: Initializing environment...', 'green'))
        for key in env_config:
            setattr(self, key, env_config[key])

        self.config = env_config

        self.action_space = gym.spaces.Box(low=np.array([-1, -params.min_blade_pitch_ratio, 0]),
                                           high=np.array([1, 1, 1]),
                                           dtype=np.float32)

        # Legal limits for state observations
        low = np.array([-np.pi,  # theta
                        -np.finfo(np.float32).max,  # theta_dot
                        -np.finfo(np.float32).max,  # omega
                        0,  # wind speed
                        ])
        high = np.array([np.pi,  # theta
                         np.finfo(np.float32).max,  # theta_dot
                         np.finfo(np.float32).max,  # omega
                         self.max_wind_speed,  # wind speed
                         ])

        self.observation_space = gym.spaces.Box(low=low,
                                                high=high,
                                                dtype=np.float32)

        ## PSF init ##
        psf_Ts = 1  # NEEDS NEW variable
        A_cont = jacobian(sym.symbolic_x_dot_simple, sym.x)
        A_disc = np.eye(3) + psf_Ts * A_cont  # Euler discretization
        B_cont = jacobian(sym.symbolic_x_dot_simple, sym.u)
        B_disc = psf_Ts * B_cont  # Euler discretization
        free_vars = SX.get_free(Function("list_free_vars", [], [A_disc, B_disc]))
        desired_seq = ["Omega", "u_p", "P_ref", "w"]
        current_seq = [a.name() for a in free_vars]
        change_to_seq = [current_seq.index(d) for d in desired_seq]
        free_vars = [free_vars[i] for i in change_to_seq]
        self.psf = PSF({"A": A_disc, "B": B_disc, "Hx": sym.Hx, "Hu": sym.Hu, "hx": sym.hx, "hu": sym.hu},
                       N=30,
                       R=np.diag([15e6 / 50000 ** 2, 15e6 / 0.349 ** 2, 1 / 15e6]),
                       PK_path=Path("PSF", "stored_PK"),
                       lin_points=free_vars,
                       lin_bounds={"w": [3 * params.wind_inflow_ratio, 25 * params.wind_inflow_ratio],
                                   "u_p": [0 * DEG2RAD, 20 * DEG2RAD],
                                   "Omega": [5 * RPM2RAD, 7.55 * RPM2RAD],
                                   "P_ref": [0, 15e6]})

        ## END PSF init ##

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

        if self.use_psf:
            action_F_thr = action[0]*params.max_thrust_force
            action_blade_pitch = action[1]*params.max_blade_pitch
            action_power = action[2]*params.max_power_generation
            action_un_normalized = [action_F_thr, action_blade_pitch, action_power]
            linearization_point = [self.turbine.omega, action_blade_pitch, action_power, self.turbine.adjusted_wind_speed]

            psf_corrected_action_un_normalized = self.psf.calc(self.turbine.state, action_un_normalized, linearization_point)

            psf_corrected_action = [psf_corrected_action_un_normalized[0]/params.max_thrust_force,
                                psf_corrected_action_un_normalized[1]/params.max_blade_pitch,
                                psf_corrected_action_un_normalized[2]/params.max_power_generation]
            self.turbine.step(psf_corrected_action, self.wind_speed)
            self.psf_action = psf_corrected_action
        else:
            self.turbine.step(action, self.wind_speed)
            self.psf_action = [0]*len(action)

        self.agent_action = action

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
        self.wind_speed = (self.max_wind_speed-self.min_wind_speed)*self.rand_num_gen.rand() + self.min_wind_speed
        self.turbine = Turbine(self.wind_speed, self.step_size)

    def calculate_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode is done.
        """
        done = False

        theta_deg = self.turbine.platform_angle*RAD2DEG
        theta_dot_deg_s = self.turbine.state[1]*RAD2DEG
        omega_rpm = self.turbine.state[2]*RAD2RPM
        power_error_MegaWatts = np.abs(action[2]-self.turbine.power_regime(self.wind_speed))*(self.turbine.max_power_generation/1e6)

        omega_ref_rpm = self.turbine.omega_setpoint(self.wind_speed)*RAD2RPM
        omega_error_rpm = np.abs(omega_rpm-omega_ref_rpm)

        self.theta_reward = np.exp(-self.gamma_theta*(np.abs(theta_deg))) - self.gamma_theta*np.abs(theta_deg)
        self.theta_dot_reward = -self.reward_theta_dot*theta_dot_deg_s**2
        self.omega_reward = np.exp(-self.gamma_omega*omega_error_rpm) - self.gamma_omega*omega_error_rpm
        self.power_reward = np.exp(-self.gamma_power*power_error_MegaWatts) - self.gamma_power*power_error_MegaWatts
        self.control_reward = -self.reward_control*(action[0]**2 + action[1]**2)

        step_reward = self.theta_reward + self.theta_dot_reward + self.omega_reward + self.power_reward + self.control_reward + self.reward_survival

        end_cond_2 = self.t_step >= self.max_episode_time/self.step_size
        crash_cond_1 = np.abs(self.turbine.platform_angle) > self.crash_angle_condition
        crash_cond_2 = self.turbine.omega > self.crash_omega_max
        crash_cond_3 = self.turbine.omega < self.crash_omega_min

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
        obs = np.hstack([self.turbine.state, self.wind_speed])
        return obs

    def seed(self, seed=None):
        """Reseeds the random number generator used in the environment"""
        self.rand_num_gen, seed = seeding.np_random(seed)
        return [seed]

    def save_latest_step(self):
        self.episode_history.setdefault('states', []).append(np.copy(self.turbine.state))
        self.episode_history.setdefault('input', []).append(self.turbine.input)
        self.episode_history.setdefault('observations', []).append(self.observation)
        self.episode_history.setdefault('time', []).append(self.t_step*self.step_size)
        self.episode_history.setdefault('last_reward', []).append(self.last_reward)
        self.episode_history.setdefault('wind_force',[]).append(self.turbine.wind_force)
        self.episode_history.setdefault('wind_torque',[]).append(self.turbine.wind_torque)
        self.episode_history.setdefault('generator_torque',[]).append(self.turbine.generator_torque)
        self.episode_history.setdefault('adjusted_wind_speed',[]).append(self.turbine.adjusted_wind_speed)

        self.episode_history.setdefault('theta_reward',[]).append(self.theta_reward)
        self.episode_history.setdefault('theta_dot_reward',[]).append(self.theta_dot_reward)
        self.episode_history.setdefault('omega_reward',[]).append(self.omega_reward)
        self.episode_history.setdefault('power_reward',[]).append(self.power_reward)
        self.episode_history.setdefault('control_reward',[]).append(self.control_reward)

        self.episode_history.setdefault('agent_actions',[]).append(self.agent_action)
        self.episode_history.setdefault('psf_actions',[]).append(self.psf_action)

    def save_latest_episode(self):
        self.history = {
            'episode_num': self.episode,
            'avg_abs_theta': np.abs(np.array(self.episode_history['states'])[:, 0]).mean(),
            'std_theta': np.array(self.episode_history['states'])[:, 0].std(),
            'crashed': int(self.crashed),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'duration': self.t_step * self.step_size,
            'wind_speed': self.wind_speed,
            'theta_reward': np.array(self.episode_history['theta_reward']).mean(),
            'theta_dot_reward': np.array(self.episode_history['theta_dot_reward']).mean(),
            'omega_reward': np.array(self.episode_history['omega_reward']).mean(),
            'power_reward': np.array(self.episode_history['power_reward']).mean(),
            'control_reward': np.array(self.episode_history['control_reward']).mean(),
        }

        self.total_history.append(self.history)
