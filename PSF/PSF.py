import itertools
import pickle
from hashlib import sha1
from pathlib import Path

import numpy as np
from casadi import SX, Function, vertcat, inf, nlpsol, jacobian, mpower, qpsol, vertsplit

LEN_FILE_STR = 20


class PSF:
    def __init__(self,
                 sys,
                 N,
                 T,
                 R=None,
                 PK_path="",
                 param=None,
                 lin_bounds=None,
                 P=None,
                 alpha=None,
                 K=None,
                 slew_rate=None,
                 LP_flag=False,
                 slack_flag=True,
                 terminal_flag=False,
                 jit_flag=False,
                 ):
        if LP_flag:
            raise NotImplementedError("Linear MPC is not implemented")
        self.jit_flag = jit_flag
        self.terminal_flag = terminal_flag
        self.slack_flag = slack_flag

        self.LP_flag = LP_flag

        self.sys = sys
        self.Ac = jacobian(self.sys["xdot"], self.sys["x"])
        self.Bc = jacobian(self.sys["xdot"], self.sys["u"])

        self.N = N
        self.T = T
        self.lin_bounds = lin_bounds

        self.PK_path = PK_path
        self.nx = self.sys["x"].shape[0]
        self.nu = self.sys["u"].shape[0]
        self.np = self.sys["p"].shape[0]

        self.slew_rate = slew_rate

        self._centroid_Px = np.zeros((self.nx, 1))
        self._centroid_Pu = np.zeros((self.nu, 1))
        self._init_guess = np.array([])

        if param is None:
            self.param = SX([])
        else:
            self.param = param

        if R is None:
            self.R = np.eye(self.nu)
        else:
            self.R = R

        if self.terminal_flag:
            if P is None:
                self.set_terminal_set()
            else:
                self.P = P
                self.alpha = alpha
                self.K = K

        self._formulate_problem()

    def _formulate_problem(self):

        #if self.LP_flag:
        #    self._set_linear_model_step()
        #else:
        self._set_nonlinear_model_step()

        X0 = SX.sym('X0', self.nx)
        X = SX.sym('X', self.nx, self.N + 1)
        U = SX.sym('U', self.nu, self.N)
        u_L = SX.sym('u_L', self.nu)
        u0 = SX.sym('u0', self.nu)
        p = SX.sym("p", self.np, 1)
        if self.slack_flag:
            eps = SX.sym("eps", self.nx, self.N)
            objective = (u_L - U[:, 0]).T @ self.R @ (u_L - U[:, 0]) + 10e9 * eps[:].T @ eps[:]
        else:
            objective = (u_L - U[:, 0]).T @ self.R @ (u_L - U[:, 0])

        w = []
        w0 = []

        g = []
        self.lbg = []
        self.ubg = []

        w += [X[:, 0]]
        w0 += [X0]

        g += [X0 - X[:, 0]]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        for i in range(self.N):
            w += [U[:, i]]
            w0 += [u0]
            # Composite Input constrains

            g += [self.sys["Hu"] @ U[:, i]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hu"]]

            w += [X[:, i + 1]]
            w0 += [self.model_step.fold(i+1).expand()(xk=X0, x_lin=X0, u=u0, u_lin=u0, p=p)["xf"]]

            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hx"]]
            if self.slack_flag:
                w += [eps[:, i]]
                w0 += [0] * self.nx
                g += [X[:, i + 1] - self.model_step(xk=X0, x_lin=X0, u=u0, u_lin=u0, p=p)['xf'] + eps[:, i]]
            else:
                g += [X[:, i + 1] - self.model_step(xk=X0, x_lin=X0, u=u0, u_lin=u0, p=p)["xf"]]
            # State propagation

            self.lbg += [0] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

        if self.slew_rate is not None:
            DT = self.T / self.N
            for i in range(self.N - 1):
                g += [U[:, i + 1] - U[:, i]]
                self.lbg += [-np.array(self.slew_rate)*DT]
                self.ubg += [np.array(self.slew_rate)*DT]

        # Terminal Set constrain
        if self.terminal_flag:
            XN_shifted = X[:, self.N] - self._centroid_Px
            g += [XN_shifted.T @ self.P @ XN_shifted - [self.alpha]]
            self.lbg += [-inf]
            self.ubg += [0]

        #if self.LP_flag:
        #    lin_points = [*vertsplit(self.sys["x"]), *vertsplit(self.sys["u"]), *vertsplit(self.sys["p"])]
        #    LP = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, u0, p, *lin_points)}
        #
        #    opts = {"osqp": {"verbose": 0, "polish": False}}
        #    self.solver = qpsol("solver", "osqp", LP, opts)
        #else:
        # JIT
        # Pick a compiler
        # compiler = "gcc"  # Linux
        # compiler = "clang"  # OSX
        compiler = "cl.exe"  # Windows

        flags = ["/O2"]  # win
        jit_options = {"flags": flags, "verbose": True, "compiler": compiler}

        # JIT
        opts = {
            "error_on_fail": True,
            "eval_errors_fatal": True,
            "verbose_init": False,
            "ipopt": {"print_level": 2},
            "print_time": False,
            "compiler": "shell",
            "jit": self.jit_flag,
            'jit_options': jit_options
        }
        NLP = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, u0, p)}
        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, u0, p)}

        self.solver = nlpsol("solver", "ipopt", prob, opts)
        self.eval_w0 = Function("eval_w0", [X0, u_L, u0, p], [vertcat(*w0)])

    def calc(self, x, u_L, u0, ext_params, reset_x0=False):
        if self._init_guess.shape[0] == 0:
            self._init_guess = np.asarray(self.eval_w0(x, u_L, u0, ext_params))

        solution = self.solver(p=vertcat(x, u_L, u0, ext_params),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               x0=self._init_guess
                               )
        if not reset_x0:
            prev = np.asarray(solution["x"])
            self._init_guess = prev

        return np.asarray(solution["x"][self.nx:self.nx + self.nu]).flatten()

    def reset_init_guess(self):
        self._init_guess = np.array([])

    def set_terminal_set(self):
        self.alpha = 0.9
        A_set, B_set = self.create_system_set()
        s = str((A_set, B_set, self.sys["Hx"], self.sys["Hu"], self.sys["hx"], self.sys["Hx"]))
        filename = sha1(s.encode()).hexdigest()[:LEN_FILE_STR]
        path = Path(self.PK_path, filename + ".dat")
        try:
            load_tuple = pickle.load(open(path, mode="rb"))
            self.K, self.P, self._centroid_Px, self._centroid_Pu = load_tuple
        except FileNotFoundError:
            print("Could not find stored KP, using MATLAB.")
            import matlab.engine

            Hx = matlab.double(self.sys["Hx"].tolist())
            hx = matlab.double(self.sys["hx"].tolist())
            Hu = matlab.double(self.sys["Hu"].tolist())
            hu = matlab.double(self.sys["hu"].tolist())

            m_A = matlab.double(np.hstack(A_set).tolist())
            m_B = matlab.double(np.hstack(B_set).tolist())

            eng = matlab.engine.start_matlab()
            eng.eval("addpath(genpath('./'))")
            P, K, centroid_Px, centroid_Pu = eng.InvariantSet(m_A, m_B, Hx, Hu, hx, hu, nargout=4)
            eng.quit()

            self.P = np.asarray(P)
            self.K = np.asarray(K)
            self._centroid_Px = np.asarray(centroid_Px)
            self._centroid_Pu = np.asarray(centroid_Pu)
            dump_tuple = (self.K, self.P, self._centroid_Px, self._centroid_Pu)
            pickle.dump(dump_tuple, open(path, "wb"))

    def create_system_set(self):
        A_set = []
        B_set = []

        free_vars = SX.get_free(Function("list_free_vars", [], [self.Ac, self.Bc]))
        bounds = [self.lin_bounds[k.name()] for k in free_vars]  # relist as given above
        eval_func = Function("eval_func", free_vars, [self.Ac, self.Bc])

        for product in itertools.product(*bounds):  # creating maximum difference
            AB_set = eval_func(*product)
            A_set.append(np.asarray(AB_set[0]))
            B_set.append(np.asarray(AB_set[1]))

        return A_set, B_set

    def _set_linear_model_step(self):

        M = 10
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

        self.model_step = Function('F',
                                   [X0, self.sys["x"], U, self.sys["u"], self.sys["p"]],
                                   [X_next],
                                   ['xk', 'x_lin', 'u', 'u_lin', 'p'],
                                   ['xf']
                                   )
    def _set_nonlinear_model_step(self):
        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function('f',
                     [self.sys["x"], self.sys["u"], self.sys["p"]],
                     [self.sys["xdot"]])
        Xk = SX.sym('Xk', self.nx)
        U = SX.sym('U', self.nu)
        P = SX.sym('P', self.np)
        X_next = Xk
        # Not used in solution, just to fit with linear MPC (DC = Dont Care)
        DCx = SX.sym('DCx', self.nx)
        DCu = SX.sym('DCu', self.nu)

        for j in range(M):
            k1 = f(X_next, U, P)
            k2 = f(X_next + DT / 2 * k1, U, P)
            k3 = f(X_next + DT / 2 * k2, U, P)
            k4 = f(X_next + DT * k3, U, P)
            X_next = X_next + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.model_step = Function('F', [Xk, DCx, U, DCu, P], [X_next], ['xk', 'x_lin', 'u', 'u_lin', 'p'], ['xf'])


if __name__ == '__main__':
    pass
