import itertools
import pickle
from hashlib import sha1
from pathlib import Path

import numpy as np
from casadi import SX, Function, vertcat, inf, nlpsol, jacobian, mpower, qpsol, vertsplit, integrator
from scipy.linalg import block_diag

from PSF.utils import nonlinear_to_linear, create_system_set, center_optimization, lift_constrain, \
    move_system, row_scale, col_scale, robust_ellipsoid, constrain_center, max_ellipsoid, NLP_OPTS, plotEllipsoid, \
    stack_Hh, ellipsoid_volume

ERROR_F_VALUE = 10e4
WARNING_F_VALUE = 10e2

LEN_FILE_STR = 20

# JIT
# How to JIT:
# https://github.com/casadi/casadi/wiki/FAQ:-how-to-perform-jit-for-function-evaluations-of-my-optimization-problem%3F

# Pick a compiler
# compiler = "gcc"  # Linux
# compiler = "clang"  # OSX
compiler = "cl.exe"  # Windows

flags = ["/O2"]  # Windows

JIT_OPTS = {
    "compiler": "shell",
    "jit": False,
    'jit_options': {"flags": flags, "verbose": True, "compiler": compiler}
}
#
NLP_OPTS = {**NLP_OPTS, **JIT_OPTS}


class PSF:
    def __init__(self,
                 sys,
                 N,
                 T,
                 t_sys,
                 R=None,
                 Q=None,
                 PK_path="",
                 param=None,
                 alpha=0.9,
                 slew_rate=None,
                 ext_step_size=1,
                 disc_method="RK",
                 LP_flag=False,
                 slack_flag=True,
                 mpc_flag=False):

        if LP_flag:
            raise NotImplementedError("Linear MPC is not implemented")

        self.slack_flag = slack_flag
        self.mpc_flag = mpc_flag

        self.LP_flag = LP_flag

        self.sys = sys
        self.t_sys = t_sys
        self.Ac = jacobian(self.sys["xdot"], self.sys["x"])
        self.Bc = jacobian(self.sys["xdot"], self.sys["u"])

        self.N = N
        self.T = T
        self.alpha = alpha

        self.PK_path = PK_path
        self.nx = self.sys["x"].shape[0]
        self.nu = self.sys["u"].shape[0]
        self.np = self.sys["p"].shape[0]

        self.slew_rate = slew_rate
        self.ext_step_size = ext_step_size

        self._centroid_Px = np.zeros((self.nx, 1))
        self._centroid_Pu = np.zeros((self.nu, 1))
        self._init_guess = np.array([])

        self.model_step = None
        self._sym_model_step = None
        self.problem = None
        self.eval_w0 = None
        self.solver = None

        if param is None:
            self.param = SX([])
        else:
            self.param = param

        self.Q = Q
        if R is None:
            self.R = np.eye(self.nu)
        else:
            self.R = R

        self.set_terminal_set()

        self.set_model_step(disc_method)
        self.formulate_problem()
        self.set_solver()

    def _get_terminal_set(self, fake=False):
        """ Under construction """

        v_c0 = center_optimization(self.sys["xdot"], self.t_sys["v"], self.t_sys["Hv"], self.t_sys["hv"])
        x_c0, u_c0, _ = np.vsplit(v_c0, [self.nx, self.nx + self.nu])


        A, B = nonlinear_to_linear(self.sys['xdot'], self.sys['x'], self.sys['u'])

        A_set, B_set = create_system_set(A, B, self.t_sys["v"], self.t_sys["Hv"], self.t_sys["hv"])
        x_c0 = np.vstack([x_c0, 0])

        outer_Hx = lift_constrain(self.sys['Hx'])
        outer_Hu = self.sys['Hu']
        outer_hx = self.sys['hx']
        outer_hu = self.sys['hu']

        Ac_set, Bc_set, Hxc, Huc, hxc, huc = move_system(A_set,
                                                         B_set,
                                                         outer_Hx,
                                                         outer_Hu,
                                                         outer_hx,
                                                         outer_hu,
                                                         x_c0,
                                                         u_c0
                                                         )

        Bs_set, B_scale, Husr, husr = row_scale(Bc_set, Huc, huc)
        Pu, _ = col_scale(np.hstack([Husr, husr]))
        Husrc, husrc = np.hsplit(Pu, [-1])
        P, K = robust_ellipsoid(Ac_set, Bs_set, Hxc, Husrc, hxc, husrc)
        K = K * B_scale[:, None]
        # Ground after lifting
        K = K[:, :-1]
        P = P[:-1, :-1]
        x_c0 = x_c0[:-1]
        Hxc = Hxc[:, :-1]
        if fake:
            P = max_ellipsoid(Hxc, hxc)

        return P, K, x_c0, u_c0

    def set_terminal_set(self):

        s = str((self.sys, self.t_sys))
        filename = sha1(s.encode()).hexdigest()[:LEN_FILE_STR]
        path = Path(self.PK_path, filename + ".dat")
        try:
            terminal_set = pickle.load(open(path, mode="rb"))
        except FileNotFoundError:
            terminal_set = self._get_terminal_set(fake=True)
            pickle.dump(terminal_set, open(path, "wb"))

        self.K, self.P, self._centroid_Px, self._centroid_Pu = terminal_set

    def set_cvodes_model_step(self):

        """
        dae = {'x': self.sys["x"], 'p': vertcat(self.sys["u"], self.sys["p"]), 'ode': self.sys["xdot"]}
        opts = {'tf': self.T / self.N, "expand": False}
        self._sym_model_step = integrator('F_internal', 'cvodes', dae, opts)

        def parse_model_step(xk, u, p, u_lin, x_lin):
            return self._sym_model_step(x0=xk, p=vertcat(u, p))

        self.model_step = parse_model_step
        """
        raise NotImplementedError("BUG. Waiting on forum answer: "
                                  "https://groups.google.com/g/casadi-users/c/hQG3zs88wVA")

    def set_taylor_model_step(self):

        M = 2
        DT = self.T / self.N
        Ad = np.eye(self.nx)
        Bd = 0
        for i in range(1, M):
            Ad += 1 / np.math.factorial(i) * mpower(self.Ac, i) * DT ** i
            Bd += 1 / np.math.factorial(i) * mpower(self.Ac, i - 1) * DT ** i

        Bd = Bd * self.Bc

        X0 = SX.sym('X0', self.nx)
        U = SX.sym('U', self.nu)
        X_next = Ad @ X0 + Bd @ U

        self._sym_model_step = Function('F',
                                        [X0, self.sys["x"], U, self.sys["u"], self.sys["p"]],
                                        [X_next],
                                        ['xk', 'x_lin', 'u', 'u_lin', 'p'],
                                        ['xf']
                                        )

        def parse_model_step(xk, u, p, u_lin, x_lin):
            return self._sym_model_step(xk=xk, x_lin=x_lin, u=u, u_lin=u_lin, p=p)

        self.model_step = parse_model_step

    def set_RK_model_step(self):
        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function('f',
                     [self.sys["x"], self.sys["u"], self.sys["p"]],
                     [self.sys["xdot"]])
        Xk = SX.sym('Xk', self.nx)
        U = SX.sym('U', self.nu)
        P = SX.sym('P', self.np)
        X_next = Xk

        for j in range(M):
            k1 = f(X_next, U, P)
            k2 = f(X_next + DT / 2 * k1, U, P)
            k3 = f(X_next + DT / 2 * k2, U, P)
            k4 = f(X_next + DT * k3, U, P)
            X_next = X_next + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        self._sym_model_step = Function('F', [Xk, U, P], [X_next], ['xk', 'u', 'p'], ['xf'])

        def parse_model_step(xk, u, p, u_lin, x_lin):
            return self._sym_model_step(xk=xk, u=u, p=p)

        self.model_step = parse_model_step

    def set_model_step(self, method_name):
        if method_name == "RK":
            self.set_RK_model_step()
        elif method_name == "taylor":
            self.set_taylor_model_step()
        elif method_name == "cvodes":
            self.set_cvodes_model_step()
        else:
            raise ValueError(f"{method_name} is not a implemented method")

    def set_solver(self):

        if self.LP_flag:
            lin_points = [*vertsplit(self.sys["x"]), *vertsplit(self.sys["u"]), *vertsplit(self.sys["p"])]

            opts = {"osqp": {"verbose": 0, "polish": False}}
            self.solver = qpsol("solver", "osqp", self.problem, opts)
        else:
            self.solver = nlpsol("solver", "ipopt", self.problem, NLP_OPTS)

    def formulate_problem(self):

        x0 = SX.sym('x0', self.nx, 1)

        X = SX.sym('X', self.nx, self.N + 1)
        x_ref = SX.sym('x_ref', self.nx, 1)

        U = SX.sym('U', self.nu, self.N)
        u_ref = SX.sym('u_ref', self.nu, 1)

        p = SX.sym("p", self.np, 1)

        u_prev = SX.sym('u_prev', self.nu, 1)

        eps = SX.sym("eps", self.nx, self.N)

        objective = self.get_objective(X=X, x_ref=x_ref, U=U, u_ref=u_ref, eps=eps)

        # empty problem
        w = []
        w0 = []

        g = []
        self.lbg = []
        self.ubg = []

        w += [X[:, 0]]
        w0 += [x0]

        g += [x0 - X[:, 0]]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        for i in range(self.N):
            w += [U[:, i]]
            w0 += [0]*w[-1].shape[0]
            # Composite Input constrains

            g += [self.sys["Hu"] @ U[:, i]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hu"]]

            w += [X[:, i + 1]]
            w0 +=[x0]


            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hx"]]
            if self.slack_flag:
                w += [eps[:, i]]
                w0 += [0] * self.nx
                g += [X[:, i + 1] - eps[:, i]*self.model_step(xk=X[:, i], x_lin=x0, u=U[:, i], u_lin=[0]*self.nu, p=p)['xf']]
            else:
                g += [X[:, i + 1] - self.model_step(xk=X[:, i], x_lin=x0, u=U[:, i], u_lin=[0]*self.nu, p=p)["xf"]]
            # State propagation

            self.lbg += [0] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

        if self.slew_rate is not None:

            g += [U[:, 0] - u_prev]
            self.lbg += [-np.array(self.slew_rate) * self.ext_step_size]
            self.ubg += [np.array(self.slew_rate) * self.ext_step_size]

            DT = self.T / self.N
            for i in range(self.N - 1):
                g += [U[:, i + 1] - U[:, i]]
                self.lbg += [-np.array(self.slew_rate) * DT]
                self.ubg += [np.array(self.slew_rate) * DT]

        # Terminal Set constrain

        XN_shifted = X[:, self.N] - self._centroid_Px
        g += [XN_shifted.T @ self.P @ XN_shifted - [self.alpha]]
        self.lbg += [-inf]
        self.ubg += [0]

        self.eval_w0 = Function("eval_w0", [x0, u_ref, p], [vertcat(*w0)])

        self.problem = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(x0, x_ref, u_ref, u_prev, p)}

    def reset_init_guess(self):
        self._init_guess = np.array([])

    def inside_terminal(self, x, u_L, ext_params):
        x0 = np.vstack(x)
        u_L = np.vstack(u_L)
        ext_params = np.vstack([ext_params])
        x1 = np.asarray(self.model_step(xk=x0, x_lin=x0, u=u_L, u_lin=u_L, p=ext_params)['xf'])
        XN_shifted = np.vstack(x) - self._centroid_Px
        no_state_violation = self.sys["Hx"] @ x1 < self.sys["hx"]
        no_input_violation = self.sys["Hu"] @ u_L < self.sys["hu"]
        inside_terminal = (XN_shifted.T @ self.P @ XN_shifted - self.alpha) < 0
        return no_state_violation.all() and no_input_violation.all() and inside_terminal.all()

    def calc(self, x, u_L, ext_params, u_prev=None, x_ref=None, reset_x0=False, ):
        if self.inside_terminal(x, u_L, ext_params) and self.mpc_flag is False:
            return u_L
        if u_prev is None and self.slew_rate is not None:
            raise ValueError("'u_prev' must be set if 'slew_rate' is given")

        if u_prev is None:
            u_prev = u_L  # Dont care, just for vertcat match
        if x_ref is None:
            x_ref = [0] * self.nx
        if self._init_guess.shape[0] == 0:
            self._init_guess = np.asarray(self.eval_w0(x, u_L, ext_params))

        solution = self.solver(p=vertcat(x, x_ref, u_L, u_prev, ext_params),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               x0=self._init_guess
                               )
        f = float(solution["f"])
        if f > ERROR_F_VALUE:
            raise RuntimeError("Function value supersedes error threshold value.")
        elif f > WARNING_F_VALUE:
            RuntimeWarning("Function value supersedes warning threshold value")
        if not reset_x0:
            prev = np.asarray(solution["x"])
            self._init_guess = prev
        u = np.asarray(solution["x"][self.nx:self.nx + self.nu]).flatten()

        return u

    def get_objective(self, U=None, eps=None, x_ref=None, X=None, u_ref=None):

        if self.mpc_flag:
            for i in range(self.N):
                objective = (x_ref - X[:, i + 1]).T @ self.Q @ (x_ref - X[:, i + 1])
                if u_ref is not None:
                    objective += (u_ref - U[:, i]).T @ self.R @ (u_ref - U[:, i])
        else:
            objective = (u_ref - U[:, 0]).T @ self.R @ (u_ref - U[:, 0])
        if self.slack_flag:
            objective += objective + 10e6 * eps[:].T @ eps[:]
        return objective


if __name__ == '__main__':
    pass
