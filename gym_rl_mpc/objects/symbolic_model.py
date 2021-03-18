from gym_rl_mpc.utils.model_params import C_1, C_4, C_2, C_3, C_5, d_r, k_r, l_r, J_r, b_d_r
from casadi import SX, vertcat, jacobian, cos, sin, Function, nlpsol, inf
import numpy as np
from gym_rl_mpc import DEFAULT_CONFIG as CONFIG
import gym_rl_mpc.utils.model_params as params

LARGE_NUM = 1e9
RPM2RAD = 1 / 60 * 2 * np.pi
DEG2RAD = 1 / 360 * 2 * np.pi

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
    params.C_1 * cos(theta) * sin(theta) + params.C_2 * sin(theta) + params.C_3 * cos(theta) * sin(theta_dot) + params.C_4 * F_thr + params.C_5 * F_wind,
    1 / params.J_r * (Q_wind - P_ref / Omega),
)

symbolic_x_dot_simple = vertcat(
    theta_dot,
    (params.C_1 + params.C_2) * theta + params.C_3 * theta_dot + params.C_4 * F_thr + params.C_5 * F_wind_simple,
    1 / params.J_r * (Q_wind_simple - 1 / Omega * P_ref),
)

# MPC constraints

# Hz * z <= hz

Hx = np.asarray([
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1]
])

hx = np.asarray([[
    CONFIG["crash_angle_condition"],
    CONFIG["crash_angle_condition"],
    45*DEG2RAD,
    45*DEG2RAD,
    -params.omega_setpoint(0),
    params.omega_setpoint(25)
]]).T

Hu = np.asarray([
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1]
])
hu = np.asarray([[
    params.max_thrust_force,
    params.max_thrust_force,
    params.min_blade_pitch_ratio*params.max_blade_pitch,
    params.max_blade_pitch,
    0,
    params.max_power_generation,
]]).T

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


def solve_initial_problem(wind, power=0.0, thruster_force=0.0):
    objective = symbolic_x_dot.T @ symbolic_x_dot

    g = []

    g += [Hx @ x - hx]

    g += [Hu[2:4, 1] @ u_p - hu[2:4]]

    prob = {'f': objective, 'x': vertcat(x, u_p), 'g': vertcat(*g), 'p': vertcat(P_ref, w, F_thr)}
    opts = {
        "verbose": False,
        "verbose_init": False,
        "ipopt": {"print_level": 1},
        "print_time": False,
    }
    solver = nlpsol("S", "ipopt", prob, opts)

    result = solver(x0=[1, 1, 1, 1], p=vertcat(power, wind, thruster_force), lbg=-inf, ubg=0)
    state = np.asarray(result["x"])[:3].flatten()
    blade_pitch = np.asarray(result["x"])[-1].flatten()
    return state, blade_pitch


if __name__ == '__main__':
    numerical_F_wind(0, 0, 3)
    numerical_Q_wind(0, 0, 3)
    numerical_x_dot([0, 0, 3], 0.0, 0.0, 0.0, 0.0)
    solve_initial_problem(wind=15, power=15e6, thruster_force=0)
