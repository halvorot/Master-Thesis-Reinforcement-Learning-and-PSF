import numpy as np
from casadi import SX, qpsol, norm_1, Function, vertcat, mtimes, nlpsol, inf, Opti, norm_2
import cvxpy as cp


class PSF:
    def __init__(self, sys, N, P=None, alpha=None, K=None, NLP_flag=False):
        self.sys = {}
        if NLP_flag:
            raise NotImplemented("NLP mpc not implemented yet")
            # self._NLP_init(sys, N, P, alpha, K)
        else:
            self._LP_init(sys, N, P, alpha, K)

    def _LP_init(self, sys, N, P=None, alpha=None, K=None):

        self._LP_set_sys(sys)

        if P is None:
            self.set_terminal_set()
        else:
            self.P = P
            self.alpha = alpha
            self.K = K

        self.N = N
        X0 = SX.sym('X0', self.nx)
        X = SX.sym('X', self.nx, self.N + 1)
        U = SX.sym('U', self.nu, self.N)
        u_L = SX.sym('u_L', self.nu)

        objective = (u_L - U[:, 0]).T@(u_L - U[:, 0])

        w = []

        self.lbw = []
        self.ubw = []

        g = []
        self.lbg = []
        self.ubg = []

        w += [X[:, 0]]
        self.lbw += [-inf]*self.nx
        self.ubw += [inf] * self.nx

        g += [X0 - X[:, 0]]
        self.lbg +=[0]  *self.nx
        self.ubg += [0] * self.nx

        for i in range(self.N):
            # Direct Input constrain

            w += [U[:, i]]

            # Composite Input constrains
            g += [self.sys["Hu"] @ U[:, i]-self.sys["hu"]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]


            w += [X[:, i + 1]]

            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1]-self.sys["hx"]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

            # State propagation
            g += [X[:, i + 1] - self.model_step(x0=X[:,i],u=U[:, i])['xf']]
            self.lbg += [0] * self.nx
            self.ubg += [0] * self.nx

        g += [X[:, self.N].T @ self.P @ X[:, self.N]-[self.alpha]]
        self.lbg += [-inf]
        self.ubg += [0]

        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L)}

        self.solver = qpsol("solver", "qpoases", prob)

    def calc(self, x, u_L):
        solution = self.solver(p=vertcat(x, u_L),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               )
        return solution["x"][self.nx:self.nx + self.nu]

    def set_terminal_set(self):

        E = cp.Variable((self.nx, self.nx), symmetric=True)
        Y = cp.Variable((self.nu, self.nx))

        objective = cp.Minimize(-cp.log_det(E))
        constraints = []

        constraints.append(
            cp.vstack([
                cp.hstack([E, (self.sys["A"] @ E + self.sys["B"] @ Y).T]),
                cp.hstack([self.sys["A"] @ E + self.sys["B"] @ Y, E])
            ]) >> 0
        )

        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=True)
        self.P = np.linalg.inv(E.value)
        self.alpha = 1
        self.K = Y.value * self.P

    def _LP_set_sys(self, sys):
        args = ["A", "B"]
        opt_min = ["umin", "xmin"]
        opt_max = ["umax", "xmax"]
        opt_comp = ["Hx", "hx", "Hu", "hu"]

        for a in args:
            self.sys[a] = sys[a]

        for opt in opt_comp:
            self.sys[opt] = sys.get(opt, [])

        if ("Hx" in self.sys) != ("hx" in self.sys):
            raise ValueError("Both Hx and hx is mutually inclusive")
        if ("Hu" in self.sys) != ("hu" in self.sys):
            raise ValueError("Both Hu and hu is mutually inclusive")

        self.nx = self.sys["A"].shape[1]
        self.nu = self.sys["B"].shape[1]
        self._set_model_step()

    def _set_model_step(self):
        x0 = SX.sym('x0', self.nx)
        u = SX.sym('u', self.nu)
        xf = self.sys["A"]@x0 + self.sys["B"] @ u
        self.model_step = Function('f', [x0, u], [xf], ['x0', 'u'], ['xf'])


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
    Hx = np.asarray([[1, 0, 0, 0],
                     [-1, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [-1, 0, -1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, -1]])

    hx = np.asarray([[1],
                     [2],
                     [1],
                     [1],
                     [2],
                     [2],
                     [1],
                     [1]])

    Hu = np.asarray([[1, 0],
                     [-1, 0],
                     [0, 1],
                     [0, -1]])

    hu = np.asarray([[2],
                     [2],
                     [2],
                     [2]])
    P = np.asarray([[1.0000, 0.0000, -0.0000, -0.0000],
                    [0.0000, 1.0000, 0.0000, -0.0000],
                    [-0.0000, 0.0000, 0.3333, 0.0000],
                    [-0.0000, -0.0000, 0.0000, 1.0000]])

    K = np.asarray([[-0.9581, -0.8393, 0.0000, - 0.0000],
                    [0, 0, -0.3185, -0.8885]])

    psf = PSF({'A': A, 'B': B, "Hx": Hx, "hx": hx, "Hu": Hu, "hu": hu}, 20, P, 0.9, K)

    print(psf.calc(x=[0, 0, 0, 0], u_L=[0, 3]))
