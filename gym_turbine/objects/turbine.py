import numpy as np
import gym_turbine.utils.state_space as ss
import gym_turbine.utils.geomutils as geom

def odesolver45(f, y, h, wind_dir):
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
    s1 = f(y, wind_dir)
    s2 = f(y + h*s1/4.0, wind_dir)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0, wind_dir)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0, wind_dir)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0, wind_dir)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0, wind_dir)
    w = y + h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


class Turbine():
    def __init__(self, step_size):
        self.state = np.zeros(22)
        self.state[3] = 10*(np.pi/180)                  # Roll initial angle
        self.state[4] = 10*(np.pi/180)                  # Pitch initial angle
        self.state[5] = ss.H*np.sin(self.state[4])      # x_tf = H*sin(theta_p)
        self.state[6] = -ss.H*np.sin(self.state[3])     # x_ts = -H*sin(theta_r)
        self.input = np.zeros(4)
        self.step_size = step_size

    def step(self, action, wind_dir):
        DVA1 = _un_normalize_dva_input(action[0])
        DVA2 = _un_normalize_dva_input(action[1])
        DVA3 = _un_normalize_dva_input(action[2])
        DVA4 = _un_normalize_dva_input(action[3])
        self.input = np.array([DVA1, DVA2, DVA3, DVA4])

        self._sim(wind_dir)

    def _sim(self, wind_dir):

        state_o5, state_o4 = odesolver45(self.state_dot, self.state, self.step_size, wind_dir)

        self.state = state_o5
        self.state[3] = geom.ssa(self.state[3])
        self.state[4] = geom.ssa(self.state[4])

    def state_dot(self, state, wind_dir):
        """
        The right hand side of the 11 ODEs governing the Trubine dyanmics. state_dot = A*state + B*F_a
        state = [q, q_dot]
        q = [x_sg, x_sw, x_hv, theta_r, theta_p, x_tf, x_ts, x_1, x_2, x_3, x_4]
        """
        state_dot = ss.A(wind_dir).dot(state) + ss.B(wind_dir).dot(self.input) + ss.W().dot(ss.F_d)

        return state_dot

    @property
    def pitch(self):
        """
        Returns the pitch angle of the turbine
        """
        return geom.ssa(self.state[4])

    @property
    def roll(self):
        """
        Returns the roll angle of the turbine
        """
        return geom.ssa(self.state[3])

    @property
    def dva_input(self):
        """
        Returns the input array to the turbine
        """
        return self.input

    @property
    def position(self):
        """
        Returns array holding the surge, sway, heave positions of the turbine
        """
        return self.state[0:3]

    @property
    def attitude(self):
        """
        Returns an array holding the roll and pitch angles of the turbine
        """
        return self.state[3:5]

def _un_normalize_dva_input(dva_input):
    dva_input = np.clip(dva_input, -1, 1)
    return dva_input*ss.dva_max
