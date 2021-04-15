import itertools

import numpy as np
from casadi import vertcat, SX, nlpsol, inf, MX, rootfinder, Function, horzcat, jacobian
from scipy.linalg import block_diag
import cvxpy as cp

NLP_OPTS = {
    "warn_initial_bounds": True,
    "error_on_fail": True,
    "eval_errors_fatal": True,
    "verbose_init": False,
    "show_eval_warnings": False,
    "ipopt": {"print_level": 2, "sb": "yes"},
    "print_time": False,
}


def constrain_center(Hz, hz):
    nz = Hz.shape[-1]
    z0 = SX.sym('z0', nz, 1)

    prob = {'f': (Hz @ z0 - hz).T @ (Hz @ z0 - hz), 'x': z0, 'g': Hz @ z0}

    solver_z = nlpsol('solver_z', 'ipopt', prob, NLP_OPTS)

    z_c0 = np.asarray(solver_z(lbg=-inf, ubg=hz)['x'])
    return z_c0


def affine_to_linear(A, B, g):
    if isinstance(A, np.ndarray):
        return num_affine_to_linear(A, B, g)
    elif isinstance(A, SX) or isinstance(A, MX):
        return sym_affine_to_linear(A, B, g)
    else:
        TypeError("Unknown type")


def sym_affine_to_linear(A, B, g):
    if A[-1, :].is_zero() and B[-1, :].is_zero():
        Al = A.__deepcopy__()
        Bl = B.__deepcopy__()
        Al[:, -1] += g
    else:
        a_shape = np.asarray(A.shape)
        a_shape += 1

        b_shape = np.asarray(B.shape)
        b_shape[0] += 1

        Al = SX(np.zeros(a_shape))
        Bl = SX(np.zeros(b_shape))
        Al[:-1, :-1] = A
        Bl[:-1, :] = B

        Al[:-1, -1] += g
    return Al, Bl


def num_affine_to_linear(Aa, Ba, g):
    if Aa[..., :].all() == 0 and Ba[..., -1, :].all() == 0:
        # The system seems already to be lifted once, adding to last row of A matrix
        Al = Aa.copy()
        Bl = Ba.copy()
        Al[..., :, -1] += g.reshape(Al[..., :, -1].shape)
    else:
        a_pad = np.tile([0, 0], (Aa.ndim, 1))
        a_pad[-2:, -1] = 1
        Al = np.pad(Aa, a_pad)

        b_pad = np.tile([0, 0], (Ba.ndim, 1))
        b_pad[-2, -1] = 1
        Bl = np.pad(Ba, b_pad)

        Al[..., :-1, -1] += g.reshape(Al[..., :-1, -1].shape)

    return Al, Bl


def move_constraint(Hz, hz, z_c0):
    return Hz, hz - Hz @ z_c0


def lift_constrain(Hx):
    if (Hx[:, -1] == 0).all():
        return Hx
    else:
        return np.hstack([Hx, np.zeros((Hx.shape[0], 1))])


def move_system(A, B, Hx, Hu, hx, hu, x_c0, u_c0):
    Hx0, hx0 = move_constraint(Hx, hx, x_c0)
    Hu0, hu0 = move_constraint(Hu, hu, u_c0)
    g = A @ x_c0 + B @ u_c0
    A0, B0 = affine_to_linear(A, B, g)
    Hx0 = lift_constrain(Hx0)
    return A0, B0, Hx0, Hu0, hx0, hu0


def row_scale(B, Hu=None, hu=None):
    axis = tuple(i for i in range(B.ndim - 1))
    B_scale = np.max(B, axis=axis) - np.min(B, axis=axis)
    if Hu is None and hu is None:
        return B / B_scale, B_scale
    else:
        return B / B_scale, B_scale, Hu / B_scale, hu


def row_un_scale(B, B_scale, Hu=None, hu=None):
    if Hu is None and hu is None:
        return B * B_scale
    else:
        return B * B_scale, Hu * B_scale, hu


def col_scale(arr=None):
    arr, scale = row_scale(np.moveaxis(arr, -2, -1))
    return np.moveaxis(arr, -2, -1), scale


def col_un_scale(arr, scale):
    arr = row_un_scale(np.moveaxis(arr, -2, -1), scale)
    return np.moveaxis(arr, -2, -1)


def center_optimization(f, v, Hv, hv, v0=None, p=None, p0=None):
    nf = f.shape[0]
    if v0 is None:
        v0 = constrain_center(Hv, hv)

    v_lwb = np.asarray([min_func(v[i], v, Hv, hv) for i in range(v.shape[0])])
    v_uwb = -np.asarray([min_func(-v[i], v, Hv, hv) for i in range(v.shape[0])])
    v_span = np.vstack(v_uwb - v_lwb)

    norm_Hh = (Hv @ v - hv) / (Hv @ v_span)

    objective = norm_Hh.T @ norm_Hh

    g = []
    lbg = []
    ubg = []

    g += [f]
    lbg += [0] * nf
    ubg += [0] * nf

    g += [Hv @ v]
    lbg += [-inf] * g[-1].shape[0]
    ubg += [hv]

    problem = {'f': objective, 'x': v, 'g': vertcat(*g)}
    if p is None:
        solver = nlpsol("solver", "ipopt", problem, NLP_OPTS)
        sol = solver(lbg=vertcat(*lbg), ubg=vertcat(*ubg), x0=v0)
    else:
        problem["p"] = p
        solver = nlpsol("solver", "ipopt", problem, NLP_OPTS)
        sol = solver(lbg=vertcat(*lbg), ubg=vertcat(*ubg), x0=v0, p=p0)

    v_c0 = np.asarray(sol['x'])

    return v_c0


def min_func(f, v, Hv, hv):
    problem = {'f': f, 'x': v, 'g': Hv @ v}
    solver = nlpsol("solver", "ipopt", problem, NLP_OPTS)
    return float(solver(lbg=-inf, ubg=hv, x0=constrain_center(Hv, hv))['f'])


def delta_range(delta, v, Hv, hv):
    min_delta = min_func(delta, v, Hv, hv)
    max_delta = min_func(-delta, v, Hv, hv)
    return min_delta, max_delta


def create_system_set(A, B, v, Hv, hv, full=True):
    nx = A.shape[0]
    delta_bounds = {}
    delta = []
    AB = horzcat(A, B)
    if full:
        AB_delta = SX(np.zeros(AB.shape))
        for row in range(AB.shape[0]):
            for col in range(AB.shape[1]):
                if AB[row, col].is_constant():
                    AB_delta[row, col] = AB[row, col]
                else:
                    delta_str = 'd_' + str(row) + str(col)
                    # print("Solving for: " + delta_str)
                    delta.append(SX.sym(delta_str))
                    AB_delta[row, col] = delta[-1]
                    delta_bounds[delta_str] = delta_range(AB[row, col], v, Hv, hv)

        eval_func = Function("eval_func", delta, [AB_delta])
    else:
        for i in range(v.shape[0]):
            delta_str = 'd_' + str(i)
            delta_bounds[delta_str] = delta_range(v[i], v, Hv, hv)
            delta.append(v[i])
        eval_func = Function("eval_func", delta, [AB])
    A_set = []
    B_set = []

    for product in itertools.product(*delta_bounds.values()):  # creating maximum difference
        AB_extreme = np.asarray(eval_func(*product))

        A_set.append(AB_extreme[:, :nx])
        B_set.append(AB_extreme[:, nx:])

    return np.stack(A_set), np.stack(B_set)


def max_ellipsoid(Hx, hx):
    nx = Hx.shape[-1]
    E = cp.Variable((nx, nx), symmetric=True)
    objective = -cp.log_det(E)
    constraints = []
    constraints.append(E >> 0)
    for j in range(Hx.shape[0]):
        constraints.append(cp.bmat([
            [hx[j, None] ** 2, Hx[j, None] @ E],
            [(Hx[j, None] @ E).T, E]
        ]) >> 0)
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver='MOSEK', verbose=False)
    return np.linalg.inv(E.value)


def robust_ellipsoid(A_set_list, B_set_list, Hx, Hu, hx, hu):
    nx = Hx.shape[-1]
    nu = Hu.shape[-1]
    E = cp.Variable((nx, nx), symmetric=True)
    Y = cp.Variable((nu, nx))

    objective = -cp.log_det(E)
    constraints = []

    for A, B in zip(A_set_list, B_set_list):
        constraints.append(E @ A.T + A @ E + Y.T @ B.T + B @ Y << 0)

    for j in range(Hx.shape[0]):
        constraints.append(cp.bmat([
            [hx[j, None] ** 2, Hx[j, None] @ E],
            [(Hx[j, None] @ E).T, E]
        ]) >> 0)

    for k in range(Hu.shape[0]):
        constraints.append(cp.bmat([
            [hu[k, None] ** 2, Hu[k, None] @ Y],
            [(Hu[k, None] @ Y).T, E]
        ]) >> 0)

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(solver="MOSEK", verbose=False)
    P = np.linalg.inv(E.value)
    K = Y.value @ P
    return P, K


def nonlinear_to_linear(x_dot, x, u):
    nabla_x = jacobian(x_dot, x)
    nabla_u = jacobian(x_dot, u)
    return affine_to_linear(nabla_x, nabla_u, x_dot - nabla_x @ x - nabla_u @ u)


def plotEllipsoid(P, Hx, hx, x0=None):
    if x0 is None:
        x0 = np.vstack([0] * P.shape[0])
    import matlab.engine
    P = matlab.double(P.tolist())
    Hx = matlab.double(Hx.tolist())
    hx = matlab.double(hx.tolist())
    x0 = matlab.double(x0.tolist())

    eng = matlab.engine.start_matlab()

    a = eng.plotEllipsoid(P, Hx, hx, x0)
    eng.quit()


def ellipsoid_volume(P):
    d, v = np.linalg.eig(P)
    volume = 4 / 3 * np.pi * np.prod(d ** (-1 / 2))
    return volume


def Hh_from_disconnected_constraints(arr):
    arr = np.asarray(arr)
    if any(arr[:, 0] > arr[:, 1]):
        raise ValueError("Upper bound must be equal or greater than lower bound")

    n = arr.shape[0]
    H = np.eye(n).repeat(2, axis=0)
    H[0::2] *= -1
    h = np.vstack(arr.flatten())
    h[0::2] *= -1
    return H, h


def stack_Hh(H_list, h_list):
    return block_diag(*H_list), np.vstack(h_list)


if __name__ == '__main__':
    """
    from gym_rl_mpc.objects.symbolic_model import *

    wind = 15

    nabla_x = jacobian(symbolic_x_dot, x)
    nabla_u = jacobian(symbolic_x_dot, u)

    A, B = nonlinear_to_linear(symbolic_x_dot, x, u)

    v = vertcat(*[x, u, w])
    Hw = np.asarray([[1], [-1]])
    hw = np.asarray([[10], [25]])
    Hv = block_diag(Hx, Hu, Hw)
    hv = np.vstack([hx, hu, hw])
    A_set, B_set = create_system_set(A, B, v, Hv, hv)

    x_c0, u_c0 = center_optimization(symbolic_x_dot, x, u, w, wind, Hx, Hu, hx, hu)
    x_c0 = np.vstack([x_c0, 0])
    Hx0 = lift_constrain(Hx)

    Ac_set, Bc_set, Hxc, Huc, hxc, huc = move_system(A_set, B_set, Hx0, Hu, hx, hu, x_c0, u_c0)

    Bs_set, B_scale, Hus, hus = row_scale(Bc_set, Huc, huc)
    Pu, _ = col_scale(np.hstack([Hus, hus]))

    Hus, hus = np.hsplit(Pu, [-1])
    P, K = robust_ellipsoid(Ac_set, Bs_set, Hxc, Hus, hxc, hus)
    P = P[:-1, :-1]
    d, v = np.linalg.eig(P)
    volume = 4 / 3 * np.pi * np.prod(d ** (-1 / 2))
    print(volume)
    print_flag = True
    P = max_ellipsoid(Hxc[:, :-1], hxc)
    if print_flag:
       
    """
    pass
