from gym_rl_mpc.utils.model_params import C_1, C_4, C_2, C_3, C_5, d_r, k_r, l_r, J_r, b_d_r
from casadi import SX, vertcat, jacobian, cos, sin, Function, nlpsol, inf
import numpy as np
from gym_rl_mpc import DEFAULT_CONFIG
import gym_rl_mpc.utils.model_params as params

LARGE_NUM = np.finfo(np.float32).max
RPM2RAD = 1 / 60 * 2 * np.pi
DEG2RAD = 1 / 360 * 2 * np.pi

config = DEFAULT_CONFIG

# Constants
w_0 = SX.sym('w')
w = 2/3 * w_0
theta = SX.sym('theta')
theta_dot = SX.sym('theta_dot')
Omega = SX.sym('Omega')
u_p = SX.sym('u_p')
P_ref = SX.sym("P_ref")
F_thr = SX.sym('F_thr')

x = vertcat(theta, theta_dot, Omega)
u = vertcat(u_p, F_thr, P_ref)

g = k_r * (cos(u_p) * w - sin(u_p) * Omega * l_r)

g_simple = k_r * (w - u_p * Omega * l_r)

F_wind = d_r * w * w + g * Omega
Q_wind = g * w - b_d_r * Omega * Omega

F_wind_simple = d_r * w * w + g * Omega
Q_wind_simple = g * w - b_d_r * Omega * Omega

# symbolic model
symbolic_x_dot = vertcat(
    theta_dot,
    C_1 * cos(theta) * sin(theta) + C_2 * sin(theta) + C_3 * cos(theta) * sin(theta_dot) + C_4 * F_thr + C_5 * F_wind,
    1 / J_r * (Q_wind - P_ref / Omega),
)

symbolic_x_dot_simple = vertcat(
    theta_dot,
    (C_1 + C_2) * theta + C_3 * theta_dot + C_4 * F_thr + C_5 * F_wind_simple,
    1 / J_r * (Q_wind_simple - 1 / Omega * P_ref),
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

hx = np.asarray([[config["crash_angle_condition"], config["crash_angle_condition"], LARGE_NUM, LARGE_NUM, -params.omega_setpoint(config["min_wind_speed"]), params.omega_setpoint(config["max_wind_speed"])]]).T

Hu = np.asarray([
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1]
])
hu = np.asarray([[0.2*params.max_blade_pitch, params.max_blade_pitch, 0, params.max_power_generation, params.max_thrust_force, params.max_thrust_force]]).T

_numerical_x_dot = Function("numerical_x_dot",
                            [x, u_p, F_thr, P_ref, w_0],
                            [symbolic_x_dot],
                            ["x", "u_p", "F_thr", "P_ref", "w_0"],
                            ["x_dot"])
_numerical_F_wind = Function("numerical_F_wind", [Omega, u_p, w_0], [F_wind], ["Omega", "u_p", "w_0"], ["F_wind"])
_numerical_Q_wind = Function("numerical_Q_wind", [Omega, u_p, w_0], [Q_wind], ["Omega", "u_p", "w"], ["Q_wind"])


def numerical_F_wind(Omega, wind, blade_pitch):
    return np.asarray(_numerical_Q_wind(Omega, blade_pitch, wind)).flatten()[0]


def numerical_Q_wind(Omega, wind, blade_pitch):
    return np.asarray(_numerical_F_wind(Omega, blade_pitch, wind)).flatten()[0]


def numerical_x_dot(state, blade_pitch, F_thr, P_ref, wind):
    return np.asarray(_numerical_x_dot(state, blade_pitch, F_thr, P_ref, wind)).flatten()


def solve_initial_problem(wind, power=0.0, thruster_force=0.0):
    objective = symbolic_x_dot.T @ symbolic_x_dot

    g = []

    g += [Hx @ x - hx]

    g += [Hu[:2, 0] @ u_p - hu[:2]]

    prob = {'f': objective, 'x': vertcat(x, u_p), 'g': vertcat(*g), 'p': vertcat(P_ref, w_0, F_thr)}
    opts = {
        "verbose": False,
        "verbose_init": False,
        "ipopt": {"print_level": 0},
        "print_time": False,
    }
    solver = nlpsol("S", "ipopt", prob, opts)

    result = solver(x0=[1, 1, 1, 1], p=vertcat(power, wind, thruster_force), lbg=-inf, ubg=0)
    state = np.asarray(result["x"])[:3]
    blade_pitch = np.asarray(result["x"])[-1]
    return state.flatten(), blade_pitch


if __name__ == '__main__':
    numerical_F_wind(0,0,3)
    numerical_Q_wind(0, 0, 3)
    numerical_x_dot([0, 0, 3], 0.0, 0.0, 0.0, 0.0)
    solve_initial_problem(15, power=15e6, thruster_force=0)
