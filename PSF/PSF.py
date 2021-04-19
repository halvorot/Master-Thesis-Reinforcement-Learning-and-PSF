import itertools
import pickle
from hashlib import sha1
from pathlib import Path
import logging

import numpy as np
from casadi import SX, Function, vertcat, inf, nlpsol, jacobian, mpower, qpsol, vertsplit, integrator, norm_1
from scipy.linalg import block_diag

from PSF.utils import nonlinear_to_linear, create_system_set, center_optimization, lift_constrain, \
    move_system, row_scale, col_scale, robust_ellipsoid, polytope_center, max_ellipsoid, NLP_OPTS, plotEllipsoid, \
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
                 ext_step_size,
                 R=None,
                 PK_path="",
                 param=None,
                 alpha=0.9,
                 slew_rate=None,
                 ):
        self.sys = sys
        self.t_sys = t_sys

        self.dt = self.get_dt_arr(ext_step_size, N, T)
        self.alpha = alpha

        self.PK_path = PK_path
        self.nx = self.sys["x"].shape[0]
        self.nu = self.sys["u"].shape[0]
        self.np = self.sys["p"].shape[0]

        self.slew_rate = slew_rate

        self.K = None
        self.P = None
        self.x_c0 = np.zeros((self.nx, 1))
        self.u_c0 = np.zeros((self.nu, 1))
        self._init_guess = np.array([])

        self.model_step = self.get_RK_model_step()

        self.problem = None
        self.eval_w0 = None
        self.solver = None

        if param is None:
            self.param = SX([])
        else:
            self.param = param

        if R is None:
            self.R = np.eye(self.nu)
        else:
            self.R = R

        self.set_terminal_set()

        self.formulate_problem()
        self.solver = nlpsol("solver", "ipopt", self.problem, NLP_OPTS)

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
            logging.warning("Creating fake terminal set")
        logging.info(f"The volume of P is {ellipsoid_volume(P)}")

        return P, K, x_c0, u_c0

    def set_terminal_set(self):

        s = str((self.sys, self.t_sys))
        filename = sha1(s.encode()).hexdigest()[:LEN_FILE_STR]
        path = Path(self.PK_path, filename + ".dat")
        try:
            logging.info(f"Trying to load pre-stored file at: {path}")
            terminal_set = pickle.load(open(path, mode="rb"))
        except FileNotFoundError:
            logging.info("Could not find stored files, creating a new one.")
            terminal_set = self._get_terminal_set(fake=True)
            pickle.dump(terminal_set, open(path, "wb"))

        self.P, self.K, self.x_c0, self.u_c0 = terminal_set

    def get_RK_model_step(self):
        M = 4  # RK4 steps per interval

        f = Function('f',
                     [self.sys["x"], self.sys["u"], self.sys["p"]],
                     [self.sys["xdot"]])
        Xk = SX.sym('Xk', self.nx)
        U = SX.sym('U', self.nu)
        P = SX.sym('P', self.np)
        X_next = Xk

        DT = SX.sym('dt')

        for j in range(M):
            k1 = f(X_next, U, P)
            k2 = f(X_next + DT / 2 * k1, U, P)
            k3 = f(X_next + DT / 2 * k2, U, P)
            k4 = f(X_next + DT * k3, U, P)
            X_next = X_next + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        model_step = Function('F', [Xk, U, P, DT], [X_next], ['xk', 'u', 'p', 'dt'], ['xf'])

        return model_step

    @staticmethod
    def get_dt_arr(first_step, N, T):

        step = (T - first_step) / (N - 1)

        return np.array([first_step] + [step] * (N - 1))

    @staticmethod
    def line(start, end, frac):
        return start + (end - start) * frac

    def formulate_problem(self):

        N = self.dt.shape[0]
        x0 = SX.sym('x0', self.nx, 1)

        X = SX.sym('X', self.nx, N + 1)

        U = SX.sym('U', self.nu, N)
        u_ref = SX.sym('u_ref', self.nu, 1)

        p = SX.sym("p", self.np, 1)

        u_prev = SX.sym('u_prev', self.nu, 1)

        eps = SX.sym("eps", self.nx, N)

        objective = self.get_objective(U=U, u_ref=u_ref, eps=eps)

        # empty problem
        w = []
        w0 = []

        g = []
        self.lbg = []
        self.ubg = []

        w += [X[:, 0]]
        w0 += [x0]

        if self.slew_rate is not None:
            g += [u_prev - U[:, 0]]
            self.lbg += [-np.array(self.slew_rate) * self.dt[0]]
            self.ubg += [np.array(self.slew_rate) * self.dt[0]]

        g += [x0 - X[:, 0]]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx


        T = self.dt.cumsum()

        for i in range(N):

            w += [U[:, i]]
            w0 += [u_prev]

            # Composite Input constrains

            g += [self.sys["Hu"] @ U[:, i]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hu"]]

            if self.slew_rate is not None:
                g += [U[:, i]-U[:, i - 1]]
                self.lbg += [-np.array(self.slew_rate) * self.dt[i]]
                self.ubg += [np.array(self.slew_rate) * self.dt[i]]

            w += [X[:, i + 1]]
            w0 += [x0]

            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hx"]]

            w += [eps[:, i]]
            w0 += [0] * w[-1].shape[0]
            g += [X[:, i + 1] - self.model_step(xk=X[:, i], u=U[:, i], p=p, dt=self.dt[i])['xf'] + eps[:, i]]

            self.lbg += [0] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

        # Terminal Set constrain

        XN_shifted = X[:, -1] - self.x_c0
        g += [XN_shifted.T @ self.P @ XN_shifted - [self.alpha]]
        self.lbg += [-inf]
        self.ubg += [0]

        self.eval_w0 = Function("eval_w0", [x0, u_prev, p], [vertcat(*w0)])

        self.problem = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(x0, u_ref, u_prev, p)}

    def reset_init_guess(self):
        self._init_guess = np.array([])

    def inside_terminal(self, x, u_L, ext_params):
        x0 = np.vstack(x)
        u_L = np.vstack(u_L)
        ext_params = np.vstack([ext_params])
        x1 = np.asarray(self.model_step(xk=x0, u=u_L, p=ext_params, dt=self.dt[0])['xf'])
        XN_shifted = np.vstack(x1) - self.x_c0
        no_state_violation = self.sys["Hx"] @ x1 < self.sys["hx"]
        no_input_violation = self.sys["Hu"] @ u_L < self.sys["hu"]
        inside_terminal = (XN_shifted.T @ self.P @ XN_shifted - self.alpha) < 0
        return no_state_violation.all() and no_input_violation.all() and inside_terminal.all()

    def calc(self, x, u_L, ext_params, u_prev=None, reset_x0=False, ):
        if self.inside_terminal(x, u_L, ext_params):
            logging.debug("Inside Terminal no need to recalculate.")
            return u_L
        if u_prev is None and self.slew_rate is not None:
            raise ValueError("'u_prev' must be set if 'slew_rate' is .")

        if u_prev is None:
            u_prev = self.u_c0
        if self._init_guess.shape[0] == 0:
            self._init_guess = np.asarray(self.eval_w0(x, u_prev, ext_params))

        solution = self.solver(p=vertcat(x, u_L, u_prev, ext_params),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               x0=self._init_guess
                               )
        f = float(solution["f"])
        logging.debug(f"Function value: {f}")
        if f > ERROR_F_VALUE:
            raise RuntimeError("Function value supersedes error threshold value.")
        elif f > WARNING_F_VALUE:
            RuntimeWarning("Function value supersedes warning threshold value")
        if not reset_x0:
            prev = np.asarray(solution["x"])
            self._init_guess = prev
        else:
            self.reset_init_guess()

        u = np.asarray(solution["x"][self.nx:self.nx + self.nu]).flatten()

        return u

    def get_objective(self, U=None, eps=None, u_ref=None):
        objective = (u_ref - U[:, 0]).T @ self.R @ (u_ref - U[:, 0])

        objective += objective + 10e6 * eps[:].T @ eps[:]
        return objective


if __name__ == '__main__':
    pass
