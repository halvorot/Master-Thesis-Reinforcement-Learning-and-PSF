import numpy as np
import gym_auv.utils.state_space as ss
import gym_auv.utils.geomutils as geom

def odesolver45(f, y, h, nu_c):
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
    s1 = f(y, nu_c)
    s2 = f(y + h*s1/4.0, nu_c)
    s3 = f(y + 3.0*h*s1/32.0 + 9.0*h*s2/32.0, nu_c)
    s4 = f(y + 1932.0*h*s1/2197.0 - 7200.0*h*s2/2197.0 + 7296.0*h*s3/2197.0, nu_c)
    s5 = f(y + 439.0*h*s1/216.0 - 8.0*h*s2 + 3680.0*h*s3/513.0 - 845.0*h*s4/4104.0, nu_c)
    s6 = f(y - 8.0*h*s1/27.0 + 2*h*s2 - 3544.0*h*s3/2565 + 1859.0*h*s4/4104.0 - 11.0*h*s5/40.0, nu_c)
    w = y + h*(25.0*s1/216.0 + 1408.0*s3/2565.0 + 2197.0*s4/4104.0 - s5/5.0)
    q = y + h*(16.0*s1/135.0 + 6656.0*s3/12825.0 + 28561.0*s4/56430.0 - 9.0*s5/50.0 + 2.0*s6/55.0)
    return w, q


class Turbine3D():
    def __init__(self, step_size):
        self.step_size = step_size

    def step(self, action, nu_c):
        self._sim(nu_c)

    def _sim(self, nu_c):
        w, q = odesolver45(self.state_dot, self.state, self.step_size, nu_c)

    def state_dot(self, state, nu_c):
        """
        The right hand side of the 12 ODEs governing the AUV dyanmics.
        """
        eta = self.state[:6]
        nu_r = self.state[6:]

        eta_dot = geom.J(eta).dot(nu_r+nu_c)
        nu_r_dot = ss.M_inv().dot(
            ss.B(nu_r).dot(self.input) - ss.D(nu_r).dot(nu_r) - ss.C(nu_r).dot(nu_r) - ss.G(eta)
        )
        state_dot = np.hstack([eta_dot, nu_r_dot])
        return state_dot

    @property
    def attitude(self):
        """
        Returns an array holding the attitude of the Turbine wrt. to NED coordinates.
        """
        return self.state[0:3]
