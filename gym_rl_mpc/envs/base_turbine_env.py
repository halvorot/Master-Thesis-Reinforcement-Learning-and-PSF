from abc import ABC, abstractmethod
from pathlib import Path

import gym
import numpy as np
from gym.utils import seeding

import gym_rl_mpc.objects.symbolic_model as sym
import gym_rl_mpc.utils.model_params as params
from PSF.PSF import PSF
from gym_rl_mpc.utils.model_params import RAD2DEG, RAD2RPM, DEG2RAD


class BaseTurbineEnv(gym.Env, ABC):
    """
    Creates an environment with a turbine.
    """

    def __init__(self, env_config):
        print('Initializing environment...')
        for key in env_config:
            setattr(self, key, env_config[key])

        self.config = env_config

        action_low = np.array(
            [
                -1,  # Scaled F_thr
                -params.min_blade_pitch_ratio,  # Scaled blade pitch from MPPT angle
                0  # Scaled P_ref
            ]
            , dtype=np.float32)

        action_high = np.array(
            [
                1,  # Scaled F_thr
                1,  # Scaled blade pitch from MPPT angle
                1  # Scaled P_ref
            ]
            , dtype=np.float32)

        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # Legal limits for state observations
        obsv_low = np.array(
            [
                -np.pi,  # theta
                -np.finfo(np.float32).max,  # theta_dot
                0,  # omega
                0,  # wind speed
            ],
            dtype=np.float32)

        obsv_high = np.array(
            [
                np.pi,  # theta
                np.finfo(np.float32).max,  # theta_dot
                np.finfo(np.float32).max,  # omega
                self.max_wind_speed,  # wind speed
            ],
            dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=obsv_low, high=obsv_high, dtype=np.float32)

        sys = {
            "xdot": sym.symbolic_x_dot,
            "x": sym.x,
            "u": sym.u,
            "p": sym.w,
            "Hx": sym.Hx,
            "hx": sym.hx,
            "Hu": sym.Hu,
            "hu": sym.hu
        }
        R = np.diag(
            [
                1 / params.max_thrust_force ** 2,
                1 / params.max_blade_pitch ** 2,
                1 / params.max_power_generation ** 2
            ])
        actuation_max_rate = [params.max_thrust_rate, params.max_blade_pitch_rate, params.max_power_rate]

        lin_bounds = {
            "w": [self.min_wind_speed * params.wind_inflow_ratio, self.max_wind_speed * params.wind_inflow_ratio],
            "u_p": [0 * params.max_blade_pitch, params.max_blade_pitch],
            "Omega": [params.omega_setpoint(self.min_wind_speed), params.omega_setpoint(self.max_wind_speed)],
            "P_ref": [0, params.max_power_generation],
            "theta": [-self.crash_angle_condition, self.crash_angle_condition],
            "theta_dot": [-45 * DEG2RAD, 45 * DEG2RAD]
        }
        ## PSF init ##
        self.psf = PSF(sys=sys, N=20, T=10, ext_step_size=self.step_size, R=R, PK_path=Path("PSF", "stored_PK"),
                       lin_bounds=lin_bounds, slew_rate=actuation_max_rate, slack_flag=True)

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
        self.psf.reset_init_guess()
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
        force_done = False

        if self.use_psf:
            F_thr = action[0] * params.max_thrust_force
            blade_pitch = action[1] * params.max_blade_pitch
            power = action[2] * params.max_power_generation
            action_un_normalized = [F_thr, blade_pitch, power]
            new_adjusted_wind_speed = params.wind_inflow_ratio * self.wind_speed - params.L * np.cos(
                self.turbine.platform_angle) * self.turbine.state[1]
            psf_params = [new_adjusted_wind_speed]
            u_stable = self.turbine.u0
            u_prev = self.turbine.input

            args = dict(x=self.turbine.state,
                        u_L=action_un_normalized,
                        u_stable=u_stable,
                        ext_params=psf_params,
                        u_prev=u_prev)
            try:
                psf_corrected_action_un_normalized = self.psf.calc(**args)
                psf_corrected_action = [psf_corrected_action_un_normalized[0] / params.max_thrust_force,
                                        psf_corrected_action_un_normalized[1] / params.max_blade_pitch,
                                        psf_corrected_action_un_normalized[2] / params.max_power_generation]
                self.psf_action = psf_corrected_action

                self.turbine.step(self.psf_action, self.wind_speed)
            except RuntimeError:
                print("Casadi failed to solve step. Using agent action. Episode done")
                force_done = True
                self.turbine.step(action, self.wind_speed)
                self.psf_action = [0] * len(action)

        else:
            self.turbine.step(action, self.wind_speed)
            self.psf_action = [0] * len(action)

        self.agent_action = action

        self.observation = self.observe()

        done, reward = self.calculate_reward(action)

        self.cumulative_reward += reward
        self.last_reward = reward

        self.save_latest_step()

        self.t_step += 1

        return self.observation, reward, (done or force_done), {}

    @abstractmethod
    def generate_environment(self):
        """
        Generates environment with a turbine and a initial wind speed
        To be implemented in extensions of BaseTurbineEnv. 
        Must set the 'turbine', 'wind_speed' attributes.
        """

    def calculate_reward(self, action):
        """
        Calculates the reward function for one time step. Also checks if the episode is done.
        """
        done = False

        # Convert variables to intuitive units
        theta_deg = self.turbine.platform_angle * RAD2DEG
        theta_dot_deg_s = self.turbine.state[1] * RAD2DEG
        omega_rpm = self.turbine.state[2] * RAD2RPM
        power_error_MegaWatts = np.abs(action[2] - self.turbine.power_regime(self.wind_speed)) * (
                self.turbine.max_power_generation / 1e6)

        omega_ref_rpm = self.turbine.omega_setpoint(self.wind_speed) * RAD2RPM
        omega_error_rpm = np.abs(omega_rpm - omega_ref_rpm)

        # Set each part of the reward
        self.theta_reward = np.exp(-self.gamma_theta * (np.abs(theta_deg))) - self.gamma_theta * np.abs(theta_deg)
        self.theta_dot_reward = -self.gamma_theta_dot * theta_dot_deg_s ** 2
        self.omega_reward = np.exp(-self.gamma_omega * omega_error_rpm) - self.gamma_omega * omega_error_rpm
        self.power_reward = np.exp(-self.gamma_power * power_error_MegaWatts) - self.gamma_power * power_error_MegaWatts
        self.input_reward = -self.gamma_input * (action[0] ** 2 + action[1] ** 2)
        if self.use_psf:
            self.psf_reward = -self.gamma_psf * np.sum(np.abs(np.subtract(self.agent_action, self.psf_action)))
        else:
            self.psf_reward = 0

        # Check if episode is done
        end_cond_2 = self.t_step >= self.max_episode_time / self.step_size
        crash_cond_1 = np.abs(self.turbine.platform_angle) > self.crash_angle_condition
        crash_cond_2 = self.turbine.omega > self.crash_omega_max
        crash_cond_3 = self.turbine.omega < self.crash_omega_min

        if end_cond_2 or crash_cond_1 or crash_cond_2 or crash_cond_3:
            done = True
        if crash_cond_1 or crash_cond_2 or crash_cond_3:
            self.crashed = True

        # Without crash reward
        step_reward = (self.theta_reward
                       + self.theta_dot_reward
                       + self.omega_reward
                       + self.power_reward
                       + self.input_reward
                       + self.psf_reward
                       + self.reward_survival)

        # Without theta and crash reward
        # step_reward = (self.theta_dot_reward
        #                + self.omega_reward
        #                + self.power_reward
        #                + self.input_reward
        #                + self.psf_reward
        #                + self.reward_survival)

        # Power only
        # step_reward = self.power_reward

        # Power and crash reward only
        # if crash_cond_1 or crash_cond_2 or crash_cond_3:
        #     step_reward = self.crash_reward
        # else:
        #     step_reward = self.power_reward
        

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
        self.episode_history.setdefault('time', []).append(self.t_step * self.step_size)
        self.episode_history.setdefault('last_reward', []).append(self.last_reward)
        self.episode_history.setdefault('wind_force', []).append(self.turbine.wind_force)
        self.episode_history.setdefault('wind_torque', []).append(self.turbine.wind_torque)
        self.episode_history.setdefault('generator_torque', []).append(self.turbine.generator_torque)
        self.episode_history.setdefault('adjusted_wind_speed', []).append(self.turbine.adjusted_wind_speed)
        self.episode_history.setdefault('wind_speed', []).append(self.wind_speed)

        self.episode_history.setdefault('theta_reward', []).append(self.theta_reward)
        self.episode_history.setdefault('theta_dot_reward', []).append(self.theta_dot_reward)
        self.episode_history.setdefault('omega_reward', []).append(self.omega_reward)
        self.episode_history.setdefault('power_reward', []).append(self.power_reward)
        self.episode_history.setdefault('input_reward', []).append(self.input_reward)
        self.episode_history.setdefault('psf_reward', []).append(self.psf_reward)

        self.episode_history.setdefault('agent_actions', []).append(self.agent_action)
        self.episode_history.setdefault('psf_actions', []).append(self.psf_action)

    def save_latest_episode(self):
        self.history = {
            'episode_num': self.episode,
            'avg_abs_theta': np.abs(np.array(self.episode_history['states'])[:, 0]).mean(),
            'std_theta': np.array(self.episode_history['states'])[:, 0].std(),
            'avg_abs_theta_dot': np.abs(np.array(self.episode_history['states'])[:, 1]).mean(),
            'crashed': int(self.crashed),
            'reward': self.cumulative_reward,
            'timesteps': self.t_step,
            'duration': self.t_step * self.step_size,
            'wind_speed': self.wind_speed,
            'theta_reward': np.array(self.episode_history['theta_reward']).mean(),
            'theta_dot_reward': np.array(self.episode_history['theta_dot_reward']).mean(),
            'omega_reward': np.array(self.episode_history['omega_reward']).mean(),
            'power_reward': np.array(self.episode_history['power_reward']).mean(),
            'input_reward': np.array(self.episode_history['input_reward']).mean(),
            'psf_reward': np.array(self.episode_history['psf_reward']).mean(),
        }

        self.total_history.append(self.history)
