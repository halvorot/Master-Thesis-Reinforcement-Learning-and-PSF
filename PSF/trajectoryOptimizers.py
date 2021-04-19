from abc import ABC

class BaseTOpti(ABC):
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

    def set_solver(self):

        if self.LP_flag:
            lin_points = [*vertsplit(self.sys["x"]), *vertsplit(self.sys["u"]), *vertsplit(self.sys["p"])]

            opts = {"osqp": {"verbose": 0, "polish": False}}
            self.solver = qpsol("solver", "osqp", self.problem, opts)
        else:


    def set_model_step(self, method_name):
        if method_name == "RK":
            self.set_RK_model_step()
        elif method_name == "taylor":
            self.set_taylor_model_step()
        elif method_name == "cvodes":
            self.set_cvodes_model_step()
        else:
            raise ValueError(f"{method_name} is not a implemented method")

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


class PSF()

