import numpy as np
import gym_rl_mpc.utils.model_params as params
import gym_rl_mpc.utils.geomutils as geom
from gym_rl_mpc.objects.symbolic_model import solve_initial_problem, numerical_F_wind, numerical_Q_wind, numerical_x_dot


def odesolver45(f, y, h, wind_speed):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 4 approx.
        w: float. Order 5 approx.
    """
    s1 = f(y, wind_speed)
    s2 = f(y + h * s1 / 4.0, wind_speed)
    s3 = f(y + 3.0 * h * s1 / 32.0 + 9.0 * h * s2 / 32.0, wind_speed)
    s4 = f(y + 1932.0 * h * s1 / 2197.0 - 7200.0 * h * s2 / 2197.0 + 7296.0 * h * s3 / 2197.0, wind_speed)
    s5 = f(y + 439.0 * h * s1 / 216.0 - 8.0 * h * s2 + 3680.0 * h * s3 / 513.0 - 845.0 * h * s4 / 4104.0, wind_speed)
    s6 = f(
        y - 8.0 * h * s1 / 27.0 + 2 * h * s2 - 3544.0 * h * s3 / 2565 + 1859.0 * h * s4 / 4104.0 - 11.0 * h * s5 / 40.0,
        wind_speed)
    w = y + h * (25.0 * s1 / 216.0 + 1408.0 * s3 / 2565.0 + 2197.0 * s4 / 4104.0 - s5 / 5.0)
    q = y + h * (16.0 * s1 / 135.0 + 6656.0 * s3 / 12825.0 + 28561.0 * s4 / 56430.0 - 9.0 * s5 / 50.0 + 2.0 * s6 / 55.0)
    return w, q


class Pendulum:
    def __init__(self, init_angle=0, init_wind_speed=3, step_size=1):
        """
            state = [theta, theta_dot, omega]^T
        """


        self.state = np.zeros(3)    # Initialize states
        # blade_pitch need scaling hence not used
        opt_state, blade_pitch = solve_initial_problem(wind=init_wind_speed,
                                                       power=params.power_regime(init_wind_speed) * params.max_power_generation,
                                                       # needs a param
                                                       thruster_force=0)
        if init_angle is not None:
            self.state[0] = init_angle

        self.state = opt_state                                    # Initialize theta
        self.input = np.array([0, 0, params.power_regime(init_wind_speed)*params.max_power_generation])  # Initialize control input
        self.step_size = step_size
        self.alpha_thr = self.step_size/(self.step_size + params.tau_thr)
        self.alpha_blade_pitch = self.step_size/(self.step_size + params.tau_blade_pitch)
        self.omega_setpoint = params.omega_setpoint
        self.power_regime = params.power_regime
        self.max_power_generation = params.max_power_generation
        self.adjusted_wind_speed = init_wind_speed

    def step(self, action, wind_speed):
        prev_F_thr = self.input[0]
        prev_blade_pitch = self.input[1]

        commanded_F_thr = _un_normalize_thrust_input(action[0])
        commanded_blade_pitch = _un_normalize_blade_pitch_input(action[1])
        power = _un_normalize_power_input(action[2])

        # Lowpass filter the thrust force and blade pitch angle
        F_thr = self.alpha_thr * commanded_F_thr + (1 - self.alpha_thr) * prev_F_thr
        blade_pitch = self.alpha_blade_pitch * commanded_blade_pitch + (1 - self.alpha_blade_pitch) * prev_blade_pitch
        # Saturate blade pitch rate
        blade_pitch = prev_blade_pitch + np.sign(blade_pitch - prev_blade_pitch) * min(
            abs(blade_pitch - prev_blade_pitch), params.max_blade_pitch_rate) * self.step_size

        self.input = np.array([F_thr, blade_pitch, power])

        # Adjust wind speed based on inflow and structure
        w = (2 / 3) * wind_speed - params.L * np.cos(self.platform_angle) * self.state[1]  # Relative axial flux w = w_0 - w_i - x_dot = (2/3)w_0 - x_dot
        self.adjusted_wind_speed = w

        self._sim(self.adjusted_wind_speed)

    def _sim(self, wind_speed):
        state_o5, state_o4 = odesolver45(self.state_dot, self.state, self.step_size, wind_speed)

        self.state = state_o5
        self.state[0] = geom.ssa(self.state[0])

    def state_dot(self, state, wind_speed):
        """
        state = [theta, theta_dot, omega]^T
        """

        F_thr = self.input[0]
        u = self.input[1]
        power = self.input[2]
        omega = self.state[2]

        F_w = numerical_F_wind(omega, wind_speed, u)
        Q_w = numerical_Q_wind(omega, wind_speed, u)
        Q_g = power / omega

        self.F_w = F_w
        self.Q_w = Q_w
        self.Q_g = Q_g

        state_dot = numerical_x_dot(state, u, F_thr, power, wind_speed)

        return state_dot


    @property
    def platform_angle(self):
        """
        Returns the angle of the platform
        """
        return geom.ssa(self.state[0])

    @property
    def omega(self):
        """
        Returns the angle of the platform
        """
        return self.state[2]

    @property
    def wind_force(self):
        """
        Returns the force on the platform from wind
        """
        return self.F_w

    @property
    def wind_torque(self):
        """
        Returns the torque on the rotor from wind
        """
        return self.Q_w

    @property
    def generator_torque(self):
        """
        Returns the torque on the rotor from wind
        """
        return self.Q_g

    @property
    def max_thrust_force(self):
        return params.max_thrust_force


def _un_normalize_thrust_input(input):
    input = np.clip(input, -1, 1)
    return input * params.max_thrust_force


def _un_normalize_blade_pitch_input(input):
    input = np.clip(input, -0.2, 1)
    return input * params.blade_pitch_max


def _un_normalize_power_input(input):
    input = np.clip(input, 0, 1)
    return input * params.max_power_generation
