import itertools
import pickle
import time
from hashlib import sha1
from pathlib import Path

import numpy as np
from casadi import SX, qpsol, Function, vertcat, inf, jacobian
import gym_rl_mpc.objects.symbolic_model as sym

LARGE_NUM = 1e9
RPM2RAD = 1 / 60 * 2 * np.pi
DEG2RAD = 1 / 360 * 2 * np.pi

LEN_FILE_STR = 20


class PSF:
    def __init__(self,
                 sys,
                 N,
                 params=None,
                 params_bounds=None,
                 P=None,
                 alpha=None,
                 K=None,
                 NLP_flag=False
                 ):
        if params is None:
            params = []

        self.sys = sys
        self.params = vertcat(*params)
        self.params_bounds = params_bounds

        self.nx = self.sys["A"].shape[1]
        self.nu = self.sys["B"].shape[1]

        if P is None:
            self.set_terminal_set()
        else:
            self.P = P
            self.alpha = alpha
            self.K = K

        self._set_model_step()

        self._LP_init(sys, N, P, alpha, K)

    def _LP_init(self, sys, N, P=None, alpha=None, K=None):

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

        g += [X[:, self.N].T @ self.P @ X[:, self.N] - [self.alpha]]
        self.lbg += [-inf]
        self.ubg += [0]

        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, self.params)}

        self.solver = qpsol("solver", "qpoases", prob, {"printLevel": "none"})

    def calc(self, x, u_L, params):
        solution = self.solver(p=vertcat(x, u_L, params),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               )

        return np.asarray(solution["x"][self.nx:self.nx + self.nu])

    def set_terminal_set(self):
        self.alpha = 0.9
        A_set, B_set = self.create_system_set()

        b = pickle.dumps((A_set, B_set))
        filename = "11b22570b3cbcd40d3ad"
        path = Path("PSF","stored_KP", filename + ".dat")
        try:
            self.K, self.P = pickle.load(open(path, mode="rb"))
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
            m_PK = eng.InvariantSet(m_A, m_B, Hx, Hu, hx, hu)
            PK = np.asarray(m_PK)
            self.P = PK[:, :self.nx]
            self.K = PK[:, : self.nx]
            pickle.dump((self.K, self.P), open(path, "wb"))

        """
        E = cp.Variable((self.nx, self.nx), symmetric=True)
        Y = cp.Variable((self.nx, self.nu))
        
        objective = cp.Minimize(-cp.log_det(E))
        constraints = []

        for A, B in zip(A_set, B_set):
            constraints.append(
                bmat([
                    [E, (A @ E + B @ Y).T],
                    [A @ E + B @ Y, E]
                ]) >> 0
            )

        for i in range(self.nx):
            constraints.append(
                bmat([
                    [hx[None, i] ** 2, Hx[None, i, :] @ E],
                    [E @ Hx[None, i, :].T, E]
                ]) >> 0
            )

        for j in range(self.nu):
            constraints.append(
                bmat([
                    [hu[None, j] ** 2, Hu[None, j, :] @ E],
                    [E @ Hu[None, j, :].T, E]
                ]) >> 0
            )

        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=True)
        self.P = np.linalg.inv(E.value)
        self.alpha = 1
        self.K = Y.value * self.P """

    def create_system_set(self):
        A_set = []
        B_set = []

        free_vars = SX.get_free(Function("list_free_vars", [], [self.sys["A"], self.sys["B"]]))
        bounds = [self.params_bounds[k.name()] for k in free_vars]  # relist as given above
        eval_func = Function("eval_func", free_vars, [self.sys["A"], self.sys["B"]])

        for product in itertools.product(*bounds):  # creating maximum difference
            AB_set = eval_func(*product)
            A_set.append(np.asarray(AB_set[0]))
            B_set.append(np.asarray(AB_set[1]))

        return A_set, B_set

    def _set_model_step(self):
        x0 = SX.sym('x0', self.nx)
        u = SX.sym('u', self.nu)
        xf = self.sys["A"] @ x0 + self.sys["B"] @ u
        self.model_step = Function('f', [x0, u], [xf], ['x0', 'u'], ['xf'])


if __name__ == '__main__':
    A = jacobian(sym.symbolic_x_dot_simple, sym.x)
    B = jacobian(sym.symbolic_x_dot_simple, sym.u)
    free_vars = SX.get_free(Function("list_free_vars", [], [A, B]))
    desired_seq = ["Omega", "u_p", "P_ref", "w"]
    current_seq = [a.name() for a in free_vars]
    change_to_seq = [current_seq.index(d) for d in desired_seq]
    free_vars = [free_vars[i] for i in change_to_seq]
    psf = PSF({"A": np.eye(3) + A, "B": B, "Hx": sym.Hx, "Hu": sym.Hu, "hx": sym.hx, "hu": sym.hu},
              N=20,
              params=free_vars,
              params_bounds={"w": [3, 25],
                             "u_p": [5 * DEG2RAD, 6 * DEG2RAD],
                             "Omega": [5 * RPM2RAD, 8 * RPM2RAD],
                             "P_ref": [1e6, 15e6]})

    print(psf.calc([0, 0, 6 * RPM2RAD], [0, 0, 15e6], vertcat(6 * RPM2RAD, 0, 15e6, 15)))
    start = time.time()
    for i in range(1000):
        psf.calc([0, 0, 6 * RPM2RAD], [0, 0, 15e6], vertcat(6 * RPM2RAD, 0, 15e6, 15))
    end = time.time()
    print(end - start)
