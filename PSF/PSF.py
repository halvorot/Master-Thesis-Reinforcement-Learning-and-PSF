import itertools
import os
import pickle
import time
from hashlib import sha1
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas
from casadi import SX, qpsol, Function, vertcat, inf, jacobian, DM, nlpsol, external
from casadi.tools import struct

import gym_rl_mpc.objects.symbolic_model as sym
from gym_rl_mpc.utils.model_params import max_thrust_force, max_blade_pitch, max_power_generation

LARGE_NUM = 1e9
RPM2RAD = 1 / 60 * 2 * np.pi
DEG2RAD = 1 / 360 * 2 * np.pi

LEN_FILE_STR = 20


class PSF:
    def __init__(self,
                 sys,
                 N,
                 T,
                 R=None,
                 PK_path="",
                 param=None,
                 slack=False,
                 lin_bounds=None,
                 P=None,
                 alpha=None,
                 K=None
                 ):

        self.slack = slack
        self.sys = sys
        self.N = N
        self.T = T

        if param is None:
            self.param = SX([])
        else:
            self.param = param

        self.PK_path = PK_path
        self.nx = self.sys["x"].shape[0]
        self.nu = self.sys["u"].shape[0]
        self.np = self.sys["p"].shape[0]
        if R is None:
            self.R = np.eye(self.nu)
        else:
            self.R = R
        # if P is None:
        #    self.set_terminal_set()
        # else:
        #    self.P = P
        #    self.alpha = alpha
        #    self.K = K

        self._set_model_step()

        self._NLP_init()
        self.init_guess = np.array([])

    def _NLP_init(self):

        X0 = SX.sym('X0', self.nx)
        X = SX.sym('X', self.nx, self.N + 1)
        U = SX.sym('U', self.nu, self.N)
        u_L = SX.sym('u_L', self.nu)
        u0 = SX.sym('u0', self.nu)
        p = SX.sym("p", self.np, 1)
        if self.slack:
            eps = SX.sym("eps", self.nx, self.N)
            objective = (u_L - U[:, 0]).T @ self.R @ (u_L - U[:, 0]) + 10e6 * eps[:].T @ eps[:]
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
            w0 += [self.model_step.fold(20).expand()(x0=X0, u=u0, p=p)["xf"]]

            # Composite State constrains
            g += [self.sys["Hx"] @ X[:, i + 1]]
            self.lbg += [-inf] * g[-1].shape[0]
            self.ubg += [self.sys["hx"]]
            if self.slack:
                w += [eps[:, i]]
                w0 += [0] * self.nx
                g += [X[:, i + 1] - (1 + eps[:, i]) * self.model_step(x0=X[:, i], u=U[:, i], p=p)['xf']]
            else:
                g += [X[:, i + 1] - self.model_step(x0=X[:, i], u=U[:, i], p=p)["xf"]]
            # State propagation

            self.lbg += [0] * g[-1].shape[0]
            self.ubg += [0] * g[-1].shape[0]

        # g += [X[:, self.N].T @ self.P @ X[:, self.N] - [self.alpha]]
        # self.lbg += [-inf]
        # self.ubg += [0]

        opts = {
            "verbose_init": False,
            "ipopt": {"print_level": 2},
            "print_time": False,
            # "compiler": "shell",
            # "jit": True,
            # 'jit_options': {"compiler": 'cl.exe'}
        }

        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g), 'p': vertcat(X0, u_L, u0, p)}

        self.solver = nlpsol("solver", "ipopt", prob, opts)
        self.eval_w0 = Function("eval_w0", [X0, u_L,u0, p], [vertcat(*w0)])

    def calc(self, x, u_L, u0, params, reset_x0=False):
        if self.init_guess.shape[0] == 0:
            self.init_guess = np.asarray(self.eval_w0(x, u_L, u0, params))

        solution = self.solver(p=vertcat(x, u_L, u0, params),
                               lbg=vertcat(*self.lbg),
                               ubg=vertcat(*self.ubg),
                               x0=self.init_guess
                               )
        if not reset_x0:
            prev = np.asarray(solution["x"])
            self.init_guess = prev

        return np.asarray(solution["x"][self.nx:self.nx + self.nu]).flatten()

    def reset_init_guess(self):
        self.init_guess = np.array([])

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

    def create_system_set(self):
        A_set = []
        B_set = []

        free_vars = SX.get_free(Function("list_free_vars", [], [self.sys["A"], self.sys["B"]]))
        bounds = [self.sys["p"][k.name()] for k in free_vars]  # relist as given above
        eval_func = Function("eval_func", free_vars, [self.sys["A"], self.sys["B"]])

        for product in itertools.product(*bounds):  # creating maximum difference
            AB_set = eval_func(*product)
            A_set.append(np.asarray(AB_set[0]))
            B_set.append(np.asarray(AB_set[1]))

        return A_set, B_set

    def _set_model_step(self):
        M = 4  # RK4 steps per interval
        DT = self.T / self.N / M
        f = Function('f',
                     [self.sys["x"], self.sys["u"], self.sys["p"]],
                     [self.sys["xdot"]])
        X0 = SX.sym('X0', self.nx)
        U = SX.sym('U', self.nu)
        P = SX.sym('P', self.np)
        X = X0
        for j in range(M):
            k1 = f(X, U, P)
            k2 = f(X + DT / 2 * k1, U, P)
            k3 = f(X + DT / 2 * k2, U, P)
            k4 = f(X + DT * k3, U, P)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.model_step = Function('F', [X0, U, P], [X], ['x0', 'u', 'p'], ['xf'])


if __name__ == '__main__':
    psf = PSF(sys={"xdot": sym.symbolic_x_dot,
                   "x": sym.x,
                   "u": sym.u,
                   "p": sym.w,
                   "Hx": sym.Hx,
                   "hx": sym.hx,
                   "Hu": sym.Hu,
                   "hu": sym.hu
                   },
              N=20,
              T=20,
              R=np.diag([
                  1 / max_thrust_force ** 2,
                  1 / max_blade_pitch ** 2,
                  1 / max_power_generation ** 2
              ]),
              slack=True,
              )
    number_of_iter = 50
    start = time.time()
    for i in range(number_of_iter):
        print(psf.calc(x=[np.random.uniform(low=-7, high=10) * DEG2RAD,
                        0,
                        np.random.uniform(low=5, high=7.5) * RPM2RAD],
                       u_L=[
                           2 * np.random.uniform(low=-max_thrust_force, high=max_thrust_force),
                           2 * np.random.uniform(low=-4 * DEG2RAD, high=max_blade_pitch),
                           2 * np.random.uniform(low=0, high=max_power_generation)
                       ],
                       u0=vertcat([500000, 0, 10e6]),
                       params=vertcat(np.random.uniform(low=3, high=25)),
                       reset_x0=True
                       )
              )
    end = time.time()
    print(f"Solved {number_of_iter} iterations in {end - start} s, [{(end - start) / number_of_iter} s/step]")
