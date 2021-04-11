import numpy as np
from casadi import vertcat, SX, nlpsol, inf, MX


def constrain_center(Hx, Hu, hx, hu):
    nx = Hx.shape[-1]
    nu = Hx.shape[-1]
    x0 = SX.sym('x0', nx, 1)
    u0 = SX.sym("u0", nu, 1)

    prob_x = {'f': (Hx @ x0 - hx).T @ (Hx @ x0 - hx), 'x': x0, 'g': Hx @ x0}
    prob_u = {'f': (Hu @ u0 - hu).T @ (Hu @ u0 - hu), 'x': u0, 'g': Hu @ u0}

    opts = {
        "warn_initial_bounds": True,
        "error_on_fail": True,
        "eval_errors_fatal": True,
        "verbose_init": False,
        "show_eval_warnings": False,
        "ipopt": {"print_level": 0, "sb": "yes"},
        "print_time": False,
    }

    solver_x = nlpsol('solver_x', 'ipopt', prob_x, opts)
    solver_u = nlpsol('solver_u', 'ipopt', prob_u, opts)

    x_c0 = np.asarray(solver_x(lbg=-inf, ubg=hx)['x'])
    u_c0 = np.asarray(solver_u(lbg=-inf, ubg=hu)['x'])

    return x_c0, u_c0


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
    else:
        a_pad = np.tile([0, 0], (Aa.ndim, 1))
        a_pad[-2:, -1] = 1
        Al = np.pad(Aa, a_pad)

        b_pad = np.tile([0, 0], (Ba.ndim, 1))
        b_pad[-2, -1] = 1
        Bl = np.pad(Ba, b_pad)

    Al[..., :-1, -1] += g.flatten()

    return Al, Bl


def move_constraint(Hz, hz, z_c0):
    return Hz, hz - Hz @ z_c0


def move_system(A, B, Hx, Hu, hx, hu, x_c0, u_c0):
    g = A @ x_c0 + B @ u_c0
    A0, B0 = affine_to_linear(A, B, g)
    Hx0, hx0 = move_constraint(Hx, hx, x_c0)
    Hu0, hu0 = move_constraint(Hu, hu, u_c0)
    return A0, B0, Hx0, Hu0, hx0, hu0


def normalize(B):
    axis = tuple(i for i in range(B.ndim - 1))
    B_scale = np.max(B, axis=axis) - np.min(B, axis=axis)
    return B / B_scale, B_scale


def un_normalize(B, B_scale):
    return B * B_scale


def terminal_set(A_set_list, B_set_list, Hx, Hu, hx, hu):
    eps = 1e-6
    A_set = np.stack(A_set_list)

    B_set_unscaled = np.stack(B_set_list)
    B_scale = np.max(B_set_unscaled, axis=(0, 1)) - np.min(B_set_unscaled, axis=(0, 1))
    B_set = B_set_unscaled / B_scale

    nx = A_set.shape[-1]
    nu = B_set.shape[-1]

    x_c0, u_c0 = constrain_center(Hx, Hu, hx, hu)

    hx0 = hx - Hx @ x_c0

    Hx0 = np.pad(Hx, ((0, 0), (0, 1)))

    hu0 = hu - Hu @ u_c0
    Hu0 = Hu

    A0 = np.dstack([A_set, -A_set @ x_c0 - B_set @ u_c0])
    A0 = np.pad(A0, ((0, 0), (0, 1), (0, 1)))

    B0 = np.pad(B_set, ((0, 0), (0, 1), (0, 0)))

    return [a for a in A0], [b for b in B0], Hx0, Hu0, hx0, hu0
