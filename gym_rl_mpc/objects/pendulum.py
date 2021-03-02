import numpy as np
import gym_rl_mpc.utils.model_params as params
import gym_rl_mpc.utils.geomutils as geom

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
        self.state = np.zeros(2*params.DoFs)    # Initialize states
        self.state[0] = init_angle
        self.input = 0                          # Initialize control input
        self.step_size = step_size
        self.t_step = 0
        self.F_d = 0

    def step(self, action):
        F = _un_normalize_input(action)
        self.input = float(F)

        self._sim()
        self.t_step += 1

    def _sim(self):

        state_o5, state_o4 = odesolver45(self.state_dot, self.state, self.step_size)

        self.state = state_o5
        self.state[0] = geom.ssa(self.state[0])

    def state_dot(self, state):
        """
        state = [theta, theta_dot]
        """
        L = params.L
        L_P = params.L_P
        k = params.k
        c = params.c
        m = params.m
        g = params.g
        J = params.J

        F_d = 0     # 1e7*np.min([self.t_step*self.step_size, 1])
        self.F_d = F_d
 
        state_dot = np.array([  state[1],
                                (1/J)*(-k*L**2*np.sin(state[0])*np.cos(state[0]) + m*g*(L/2)*np.sin(state[0]) - c*L*np.cos(state[0])*state[1] + self.input*L_P*np.cos(state[0]) + F_d)
                                ])

        return state_dot

    @property
    def angle(self):
        """
        Returns the angle of the pendulum
        """
        return geom.ssa(self.state[0])

    @property
    def disturbance_force(self):
        """
        Returns the magnitude of the disturbance force
        """
        return self.F_d

    @property
    def max_input(self):
        return params.max_input

def _un_normalize_input(input):
    input = np.clip(input, -1, 1)
    return input*params.max_input
