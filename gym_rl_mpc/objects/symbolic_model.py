import numpy as np
from casadi import SX, vertcat, cos, sin, Function

import gym_rl_mpc.utils.model_params as params
from PSF.utils import center_optimization, Hh_from_disconnected_constraints
from ..utils.model_params import RPM2RAD, DEG2RAD

# Constants
w = SX.sym('w')
theta = SX.sym('theta')
theta_dot = SX.sym('theta_dot')
Omega = SX.sym('Omega')
u_p = SX.sym('u_p')
P_ref = SX.sym("P_ref")
F_thr = SX.sym('F_thr')

x = vertcat(theta, theta_dot, Omega)
u = vertcat(F_thr, u_p, P_ref)

g = params.k_r * (cos(u_p) * w - sin(u_p) * Omega * params.l_r)

g_simple = params.k_r * (w - u_p * Omega * params.l_r)

F_wind = params.d_r * w * w + g * Omega
Q_wind = g * w - params.b_d_r * Omega * Omega

F_wind_simple = params.d_r * w * w + g * Omega
Q_wind_simple = g * w - params.b_d_r * Omega * Omega

# symbolic model
symbolic_x_dot = vertcat(
    theta_dot,
    params.C_1 * cos(theta) * sin(theta) + params.C_2 * sin(theta) + params.C_3 * cos(theta) * sin(
        theta_dot) + params.C_4 * F_thr + params.C_5 * F_wind,
    1 / params.J_r * (Q_wind - P_ref / Omega),
)

# Constraints

sys_lub_x = np.asarray([
    [-10 * DEG2RAD, 10 * DEG2RAD],
    [-10 * DEG2RAD, 10 * DEG2RAD],
    [5 * RPM2RAD, 7.6 * RPM2RAD]
])

sys_lub_u = np.asarray([
    [-params.max_thrust_force, params.max_thrust_force],
    [-params.min_blade_pitch_ratio * params.max_blade_pitch, params.max_blade_pitch],
    [0, params.max_power_generation]
])

sys_lub_p = np.asarray([
    [10, 25],
])

_numerical_x_dot = Function("numerical_x_dot",
                            [x, u_p, F_thr, P_ref, w],
                            [symbolic_x_dot],
                            ["x", "u_p", "F_thr", "P_ref", "w"],
                            ["x_dot"])
_numerical_F_wind = Function("numerical_F_wind", [Omega, u_p, w], [F_wind], ["Omega", "u_p", "w"], ["F_wind"])
_numerical_Q_wind = Function("numerical_Q_wind", [Omega, u_p, w], [Q_wind], ["Omega", "u_p", "w"], ["Q_wind"])


def numerical_F_wind(rotation_speed, wind, blade_pitch):
    return np.asarray(_numerical_Q_wind(rotation_speed, blade_pitch, wind)).flatten()[0]


def numerical_Q_wind(rotation_speed, wind, blade_pitch):
    return np.asarray(_numerical_F_wind(rotation_speed, blade_pitch, wind)).flatten()[0]


def numerical_x_dot(state, blade_pitch, propeller_thrust, power, wind):
    return np.asarray(_numerical_x_dot(state, blade_pitch, propeller_thrust, power, wind)).flatten()


def get_sys():
    Hx, hx = Hh_from_disconnected_constraints(sys_lub_x)

    Hu, hu = Hh_from_disconnected_constraints(sys_lub_u)

    Hp, hp = Hh_from_disconnected_constraints(sys_lub_p)

    sys = {
        "xdot": symbolic_x_dot,
        "x": x,
        "u": u,
        "p": w,
        "Hx": Hx,
        "hx": hx,
        "Hu": Hu,
        "hu": hu,
        "Hp": Hp,
        "hp": Hp
    }
    return sys


# PSF Terminal constraints
def get_terminal_sys():
    sys = get_sys()

    term_lub_x = np.asarray([
        [0 * DEG2RAD, 8 * DEG2RAD],
        [-1 * DEG2RAD, 1 * DEG2RAD],
        [6 * RPM2RAD, 7 * RPM2RAD]
    ])

    term_lub_u = np.asarray([
        [-params.max_thrust_force, params.max_thrust_force],
        [0, params.max_blade_pitch],
        [0, params.max_power_generation]
    ])
    # Wind speed

    term_lub = np.vstack((term_lub_x, term_lub_u, sys_lub_p))

    term_Hv, term_hv = Hh_from_disconnected_constraints(term_lub)

    t_sys = {
        'v': vertcat(*[sys["x"], sys["u"], sys["p"]]),
        'Hv': term_Hv,
        'hv': term_hv
    }
    return t_sys


def solve_initial_problem(wind):
    Hz, hz = Hh_from_disconnected_constraints(np.vstack([sys_lub_x, sys_lub_u]))
    z0 = center_optimization(symbolic_x_dot, vertcat(x, u), Hz, hz, p=w, p0=wind)
    x0, u0 = np.vsplit(z0, [x.shape[0]])
    return x0, u0


if __name__ == '__main__':
    solve_initial_problem(wind=25 * 2 / 3)
