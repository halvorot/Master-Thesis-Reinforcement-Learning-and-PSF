import itertools
import pickle
import time
from hashlib import sha1
from pathlib import Path

import numpy as np
from casadi import SX, qpsol, Function, vertcat, inf, jacobian, DM
import gym_rl_mpc.objects.symbolic_model as sym

LARGE_NUM = 1e9
RPM2RAD = 1 / 60 * 2 * np.pi
DEG2RAD = 1 / 360 * 2 * np.pi

LEN_FILE_STR = 20


class PSF:
    def __init__(self,
                 sys,
                 N,
                 R=None,
                 PK_path="",
                 lin_points=None,
                 lin_bounds=None,
                 P=None,
                 alpha=None,
                 K=None
                 ):
        if lin_points is None:
            lin_points = []

        self.sys = sys
        self.N = N
        self.PK_path = PK_path
        self.lin_points = vertcat(*lin_points)
        self.lin_bounds = lin_bounds

        self.nx = self.sys["A"].shape[1]
        self.nu = self.sys["B"].shape[1]
        if R is None:
            self.R = np.eye(self.nu)
        else:
            self.R = R

        if P is None:
            self.set_terminal_set()
        else:
            self.P = P
            self.alpha = alpha
            self.K = K

        self._set_model_step()

        self._LP_init(sys)

    def _LP_init(self, N):

        X0 = SX.sym('X0', self.nx)
        X = SX.sym('X', self.nx, self.N + 1)
        U = SX.sym('U', self.nu, self.N)
        u_L = SX.sym('u_L', self.nu)

        objective = (u_L - U[:, 0]).T @ self.R @ (
                u_L - U[:, 0])

        w = []

        self.lbw = []
        self.ubw = []

        g = []
        self.lbg = []
        self.ubg = []

        w += [X[:, 0]]
        self.lbw += [-inf] * self.nx
        self.ubw += [inf] * self.nx
        g += [self.sys["Hx"] @ X[:, 0]]
        self.lbg += [-inf] * g[-1].shape[0]
        self.ubg += [self.sys["hx"]]
        g += [X0 - X[:, 0]]
        self.lbg += [0] * self.nx
        self.ubg += [0] * self.nx

        for i in range(self.N):
            w += [U[:, i]]
            # Composite Input constrains

            g += [self.sys["Hu"] @ U[:, i]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hu"]]

            w += [X[:, i + 1]]

            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1] ]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hx"]]

            # State propagation
            g += [X[:, i + 1] - self.model_step(x0=X[:, i], u=U[:, i])['xf']]
            self.lbg += [0] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

        g += [X[:, self.N].T @ self.P @ X[:, self.N] - [self.alpha]]
        self.lbg += [-inf]
        self.ubg += [0]

        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, self.lin_points)}
        opts = {"verbose": False, 'warm_start_primal': True, "warm_start_dual": True,
                "osqp": {"verbose": False, "polish": True, "eps_prim_inf": 1.00e-22, "rho": 1.00e-22}}
        #self.solver = qpsol("solver", "osqp", prob, opts)
        self.solver = qpsol("solver", "qpoases", prob, {"printLevel":"none"})

    def calc(self, x, u_L, lin_dict):

        solution = self.solver(p=vertcat(x, u_L, lin_dict),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               )

        return np.asarray(solution["x"][self.nx:self.nx + self.nu]).flatten()

    def set_terminal_set(self):
        self.alpha = 0.9
        A_set, B_set = self.create_system_set()
        s = str((A_set, B_set, self.sys["Hx"], self.sys["Hu"], self.sys["hx"], self.sys["Hx"]))
        filename = sha1(s.encode()).hexdigest()[:LEN_FILE_STR]
        path = Path(self.PK_path, filename + ".dat")
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
            eng.eval("addpath(genpath('./'))")
            m_PK = eng.InvariantSet(m_A, m_B, Hx, Hu, hx, hu)
            eng.quit()
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
        bounds = [self.lin_bounds[k.name()] for k in free_vars]  # relist as given above
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
    A_cont = jacobian(sym.symbolic_x_dot_simple, sym.x)
    A_disc = np.eye(3) + A_cont  # Euler discretization
    B_cont = jacobian(sym.symbolic_x_dot_simple, sym.u)
    B_disc = B_cont  # Euler discretization
    free_vars = SX.get_free(Function("list_free_vars", [], [A_disc, B_disc]))
    desired_seq = ["Omega", "u_p", "P_ref", "w"]
    current_seq = [a.name() for a in free_vars]
    change_to_seq = [current_seq.index(d) for d in desired_seq]
    free_vars = [free_vars[i] for i in change_to_seq]
    psf = PSF({"A": A_disc, "B": B_disc, "Hx": sym.Hx, "Hu": sym.Hu, "hx": sym.hx, "hu": sym.hu},
              N=30,
              R=np.diag([15e6/ 50000**2, 15e6 / 0.349**2, 1 / 15e6]),
              PK_path="stored_PK",
              lin_points=free_vars,
              lin_bounds={"w": [3*2/3, 25*2/3],
                          "u_p": [0 * DEG2RAD, 20 * DEG2RAD],
                          "Omega": [5 * RPM2RAD, 7.55 * RPM2RAD],
                          "P_ref": [0, 15e6]})

    print(psf.calc([0, 0, 7.55 * RPM2RAD], [0, -0.1, 15e6], vertcat(7 * RPM2RAD, 0, 15e6, 12)))
    number_of_iter = 1000
    start = time.time()
    for i in range(number_of_iter):
        psf.calc([0, 0, 7.55 * RPM2RAD], [0, -0.1, 15e6], vertcat(7 * RPM2RAD, 0, 15e6, 12))
    end = time.time(
    print(f"Solved {number_of_iter} iterations in {end - start} s, [{(end - start) / number_of_iter} s/step]"))
