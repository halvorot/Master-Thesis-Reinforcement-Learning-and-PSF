import numpy as np
from casadi import vertcat, SX, nlpsol, inf


def terminal_set(A_set_list, B_set_list, Hx, Hu, hx, hu):
    eps = 1e-6
    A_set = np.stack(A_set_list)

    B_set_unscaled = np.stack(B_set_list)
    B_scale = np.max(B_set_unscaled, axis=(0, 1)) - np.min(B_set_unscaled, axis=(0, 1))
    B_set = B_set_unscaled / B_scale

    nx = A_set.shape[-1]
    nu = B_set.shape[-1]

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

    Hx_center = np.asarray(solver_x(lbg=-inf, ubg=hx)['x'])
    Hu_center = np.asarray(solver_u(lbg=-inf, ubg=hu)['x'])

    hx0 = hx - Hx @ Hx_center

    Hx0 = np.pad(Hx, ((0, 0), (0, 1)))

    hu0 = hu - Hu @ Hu_center
    Hu0 = Hu

    A0 = np.dstack([A_set, -A_set @ Hx_center - B_set @ Hu_center])
    A0 = np.pad(A0, ((0, 0), (0, 1), (0, 0)))

    B0 = np.pad(B_set, ((0, 0), (0, 1), (0, 0)))

    return [a for a in A0], [b for b in B0], Hx0, Hu0, hx0, hu0
