import numpy as np
from casadi import SX, qpsol, norm_1, Function, vertcat, mtimes, nlpsol


class PSF:
    def __init__(self, sys, N, P=None, alpha=None, K=None, NLP_flag=False):
        if NLP_flag:
            raise NotImplemented("NLP mpc not implemented yet")
            # self._NLP_init(sys, N, P, alpha, K)
        else:
            self._LP_init(sys, N, P, alpha, K)

    def _LP_init(self, sys, N, P=None, alpha=None, K=None):
        self._LP_set_sys(sys)

        self.N = N
        X0 = SX.sym('X0', self.nx)
        X = SX.sym('X', self.nx, self.N + 1)
        U = SX.sym('U', self.nu, self.N)
        u_L = SX.sym('u_L',self.nu)
        eps = SX.sym('eps')

        if P is None:
            self.set_terminal_set()
        else:
            self.P = P
            self.alpha = alpha
            self.K = K

        w = []
        w0 = []

        objective = norm_1(u_L - U[:, 0])
        g = [X0 - X[:, 0]]
        w = [X[:, 0]]
        for i in range(self.N):
            # State propagation
            w += [U[:, i]]
            w += [X[:, i + 1]]
            g += [X[:, i + 1] - self.discrete_step(x0=X[:, i], u=U[:, i])['xf']]

        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L)}
        solver =  nlpsol('S', 'ipopt', prob)

        solver(p=vertcat([0, 0, 0, 1], [0.2, 0]))
    def set_terminal_set(self):
        pass

    def _LP_set_sys(self, sys):
        self.sys = sys
        self.nx = self.sys["A"].shape[1]
        self.nu = self.sys["B"].shape[1]
        x0 = SX.sym('x0', self.nx)
        u = SX.sym('u', self.nu)
        xf = mtimes(SX(self.sys["A"]), x0) + mtimes(SX(self.sys["B"]), u)
        self.discrete_step = Function('f', [x0, u], [xf], ['x0', 'u'], ['xf'])


if __name__ == '__main__':
    Ts = 0.1
    A = np.asarray([[1, Ts, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, Ts],
                    [0, 0, 0, 1]])

    B = np.asarray([[Ts * Ts * 0.5, 0],
                    [Ts, 0],
                    [0, Ts * Ts * 0.5],
                    [0, Ts]])

    PSF({'A': A, 'B': B}, 20)
