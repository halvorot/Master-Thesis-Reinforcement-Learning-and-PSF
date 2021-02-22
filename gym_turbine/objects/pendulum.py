import numpy as np
import gym_turbine.utils.state_space_pendulum as ss
import gym_turbine.utils.geomutils as geom

def odesolver45(f, y, h):
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
    s1 = f(y)
    s2 = f(y + h*s1/4.0)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0)
    w = y + h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


class Pendulum():
    def __init__(self, init_angle, step_size):
        self.state = np.zeros(2*ss.DoFs)                # Initialize states
        self.state[0] = init_angle
        self.input = np.zeros(2)                        # Initialize control input
        self.step_size = step_size

    def step(self, action):
        Fa_1 = _un_normalize_dva_input(action[0])
        Fa_2 = _un_normalize_dva_input(action[1])
        self.input = np.array([Fa_1, Fa_2])

        self._sim()

    def _sim(self):

        state_o5, state_o4 = odesolver45(self.state_dot, self.state, self.step_size)

        self.state = state_o5
        self.state[0] = geom.ssa(self.state[0])

    def state_dot(self, state):
        """
        The right hand side of the 9 ODEs governing the Trubine dyanmics. state_dot = A*state + B*F_a + W*F_d + W_g*F_g
        state = [q, q_dot]
        q = [theta, x_1, x_2]
        """
        state_dot = ss.A().dot(state) + ss.B().dot(self.input) + ss.W().dot(ss.F_d) + ss.W_g().dot(ss.F_g)

        return state_dot

    @property
    def pitch(self):
        """
        Returns the pitch angle of the pendulum
        """
        return geom.ssa(self.state[0])


    @property
    def dva_displacement(self):
        """
        Returns array of displacements of DVAs [x_1, x_2]
        """
        return self.state[1:3]

    @property
    def dva_displacement_dot(self):
        """
        Returns array of time derivative of displacements of DVAs [x_1_dot, x_2_dot]
        """
        return self.state[4:6]

    @property
    def max_input(self):
        return ss.max_input

def _un_normalize_dva_input(dva_input):
    dva_input = np.clip(dva_input, -1, 1)
    return dva_input*ss.max_input
