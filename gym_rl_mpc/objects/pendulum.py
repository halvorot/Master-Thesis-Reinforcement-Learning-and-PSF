import numpy as np
import gym_rl_mpc.utils.model_params as params
import gym_rl_mpc.utils.geomutils as geom

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
    s2 = f(y + h*s1/4.0, wind_speed)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0, wind_speed)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0, wind_speed)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0, wind_speed)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0, wind_speed)
    w = y + h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


class Pendulum():
    def __init__(self, init_angle, init_wind_speed, step_size):
        self.state = np.zeros(3)                                        # Initialize states
        self.state[0] = init_angle                                      # Initialize theta
        self.state[2] = params.lambda_star*init_wind_speed/params.R     # Initialize Omega
        self.input = np.zeros(2)                                        # Initialize control input
        self.step_size = step_size
        self.alpha_thr = self.step_size/(self.step_size + params.tau_thr)
        self.alpha_blade_pitch = self.step_size/(self.step_size + params.tau_blade_pitch)
        self.omega_setpoint = params.omega_setpoint

    def step(self, action, wind_speed):
        prev_F_thr = self.input[0]
        prev_blade_pitch = self.input[1]

        commanded_F_thr = _un_normalize_thrust_input(action[0])
        commanded_blade_pitch = _un_normalize_blade_pitch_input(action[1])
        # Lowpass filter the thrust force and blade pitch angle
        F_thr = self.alpha_thr*commanded_F_thr + (1-self.alpha_thr)*prev_F_thr
        blade_pitch = self.alpha_blade_pitch*commanded_blade_pitch + (1-self.alpha_blade_pitch)*prev_blade_pitch
        # Saturate blade pitch rate
        blade_pitch = prev_blade_pitch + np.sign(blade_pitch-prev_blade_pitch)*min(abs(blade_pitch-prev_blade_pitch), params.max_blade_pitch_rate)*self.step_size

        self.input = np.array([F_thr, blade_pitch])

        self._sim(wind_speed)

    def _sim(self, wind_speed):

        state_o5, state_o4 = odesolver45(self.state_dot, self.state, self.step_size, wind_speed)

        self.state = state_o5
        self.state[0] = geom.ssa(self.state[0])

    def state_dot(self, state, wind_speed):
        """
        state = [theta, theta_dot, omega]^T
        """
        L_thr = params.L_thr
        k_r = params.k_r
        l_r = params.l_r
        b_d_r = params.b_d_r
        d_r = params.d_r
        J_r = params.J_r

        F_thr = self.input[0]
        u = self.input[1]
        theta = state[0]
        theta_dot = state[1]
        omega = state[2]

        F_w = d_r*np.abs(wind_speed)*wind_speed + k_r*np.cos(u)*wind_speed*omega - k_r*l_r*np.sin(u)*omega**2
        Q_w = k_r*np.cos(u)*wind_speed**2 - k_r*np.sin(u)*omega*wind_speed*l_r - b_d_r*np.abs(omega)*omega
        Q_g = params.power_regime(wind_speed)/omega

        self.F_w = F_w
        self.Q_w = Q_w
        self.Q_g = Q_g

        state_dot = np.array([  state[1],
                                4.4500746068705328*np.sin(theta)*np.cos(theta) - 4.488263864070078*np.sin(theta) - 0.0055491593253495*np.cos(theta)*theta_dot - 6.86458290065766e-12*L_thr*F_thr + 0.000000000991589*F_w,
                                (Q_w - Q_g)/J_r
                                ])

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
    return input*params.max_thrust_force

def _un_normalize_blade_pitch_input(input):
    input = np.clip(input, -0.1, 1)
    return input*params.blade_pitch_max
