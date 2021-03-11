import numpy as np
from casadi import SX, qpsol, norm_1, Function, vertcat, mtimes, nlpsol, inf, Opti, norm_2, horzcat, jacobian
import cvxpy as cp
from gym_rl_mpc.utils.model_params import J_r, k_r, l_r, b_d_r, C_1, C_2, C_3, C_5, C_4, d_r

LARGE_NUM = 1e9


class PSF:
    def __init__(self, sys, N, lin_params=None, params=None, P=None, alpha=None, K=None, NLP_flag=False
                 ):
        if params is None:
            params = []
        if lin_params is None:
            lin_params = []
        self.sys = {}
        self.lin_params = lin_params
        self.params = params
        if NLP_flag:
            raise NotImplemented("NLP mpc not implemented yet")
            # self._NLP_init(sys, N, P, alpha, K)
        else:
            self._LP_init(sys, N, P, alpha, K)

    def _LP_init(self, sys, N, P=None, alpha=None, K=None):

        self._LP_set_sys(sys)

        if not P is None:
            pass
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

        objective = (u_L - U[:, 0]).T @ (u_L - U[:, 0])

        w = []

        self.lbw = []
        self.ubw = []

        g = []
        self.lbg = []
        self.ubg = []

        w += [X[:, 0]]
        self.lbw += [-inf] * self.nx
        self.ubw += [inf] * self.nx

        g += [X0 - X[:, 0]]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        for i in range(self.N):
            # Direct Input constrain

            w += [U[:, i]]

            # Composite Input constrains
            g += [self.sys["Hu"] @ U[:, i] - self.sys["hu"]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

            w += [X[:, i + 1]]

            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1] - self.sys["hx"]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

            # State propagation
            g += [X[:, i + 1] - self.model_step(x0=X[:, i], u=U[:, i])['xf']]
            self.lbg += [0] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

        # g += [X[:, self.N].T @ self.P @ X[:, self.N] - [self.alpha]]
        # self.lbg += [-inf]
        # self.ubg += [0]

        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, self.lin_params, self.params)}

        self.solver = qpsol("solver", "qpoases", prob)

    def calc(self, x, u_L, lin_params, params):
        solution = self.solver(p=vertcat(x, u_L,lin_params, params),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               )

        return np.asarray(solution["x"][self.nx:self.nx + self.nu])

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
        xf = self.sys["A"] @ x0 + self.sys["B"] @ u
        self.model_step = Function('f', [x0, u], [xf], ['x0', 'u'], ['xf'])


if __name__ == '__main__':
    # Constants
    w = SX.sym('w')
    theta = SX.sym('theta')
    theta_dot = SX.sym('theta')
    Omega = SX.sym('Omega')
    u_p = SX.sym('u')
    P_ref = SX.sym("P_ref")
    F_thr = SX.sym('F_thr')

    x = vertcat(theta, theta_dot, Omega)
    u = vertcat(u_p, P_ref, F_thr)

    x_dot = vertcat(
        theta_dot,
        (C_1 + C_2) * theta + C_3 * theta_dot + C_4 * F_thr + C_5 * d_r * w ** 2 + k_r * (
                w * Omega - u_p * Omega ** 2 * l_r),
        1 / J_r * (k_r * (w ** 2 - u_p * Omega * w * l_r) - b_d_r * Omega ** 2 - 1 / Omega * P_ref)
    )
    A = jacobian(x_dot, x)
    B = jacobian(x_dot, u)

    Hx = np.asarray([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1]
    ])

    hx = np.asarray([[-10, 10, -LARGE_NUM, LARGE_NUM, 5, 7.56]]).T

    Hu = np.asarray([
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1]
    ])
    hu = np.asarray([[-4, 20, 0, 15e6, -5e5, 5e5]]).T

    sys = {'A': A, 'B': B, "Hx": Hx, "hx": hx, "Hu": Hu, "hu": hu}

    psf = PSF(sys, 20, lin_params=vertcat(Omega, u_p, P_ref), params=vertcat(w))

    psf.calc([0, 0, 6], [0, 15e6, 6], vertcat(5, 12, 0), vertcat(5))
