import multiprocessing

import numpy as np
from casadi import SX, vertcat, cos, sin, Function
from matplotlib import pyplot as plt

import gym_rl_mpc.utils.model_params as params
from PSF.utils import center_optimization, Hh_from_disconnected_constraints, steady_state, sample_inside_polytope
from gym_rl_mpc.utils.model_params import RPM2RAD, DEG2RAD

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
    1 / params.J_r * (Q_wind - P_ref / (Omega + 1e-6)),
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


def get_sys(
        custom_sys_lub_x=sys_lub_x,
        custom_sys_lub_u=sys_lub_u,
        custom_sys_lub_p=sys_lub_p
):
    Hx, hx = Hh_from_disconnected_constraints(custom_sys_lub_x)

    Hu, hu = Hh_from_disconnected_constraints(custom_sys_lub_u)

    Hp, hp = Hh_from_disconnected_constraints(custom_sys_lub_p)

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
        "hp": hp
    }
    return sys


# PSF Terminal constraints
def get_terminal_sys(
        term_lub_x=np.asarray([
            [0 * DEG2RAD, 8 * DEG2RAD],
            [-1 * DEG2RAD, 1 * DEG2RAD],
            [6 * RPM2RAD, 7 * RPM2RAD]
        ]),
        term_lub_u=np.asarray([
            [-params.max_thrust_force, params.max_thrust_force],
            [0, params.max_blade_pitch],
            [0, params.max_power_generation]
        ]),
        term_lub_p=sys_lub_p
):
    term_lub = np.vstack((term_lub_x, term_lub_u, term_lub_p))

    term_Hv, term_hv = Hh_from_disconnected_constraints(term_lub)

    sys = get_sys()
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


def change_power(wind, power):
    Hz, hz = Hh_from_disconnected_constraints(np.vstack([sys_lub_x, sys_lub_u[:-1]]))
    v0 = sample_inside_polytope(Hz, hz)
    print(f"\nWind:\t{wind}")
    try:
        return steady_state(symbolic_x_dot,
                            vertcat(x, F_thr, u_p),
                            Hz, hz,
                            v0=v0, p=vertcat(w, P_ref), p0=vertcat(wind, power))
    except RuntimeError:
        return np.vstack([np.nan] * Hz.shape[-1])


def change_blade_pitch(wind, pitch):
    Hz, hz = Hh_from_disconnected_constraints(np.vstack([sys_lub_x, sys_lub_u[[0, 2]]]))
    v0 = sample_inside_polytope(Hz, hz)
    # print(f"\nWind:\t{wind}")
    try:
        return steady_state(symbolic_x_dot,
                            vertcat(x, F_thr, P_ref),
                            Hz, hz,
                            v0=v0, p=vertcat(w, u_p), p0=vertcat(wind, pitch))
    except RuntimeError:
        return np.vstack([np.nan] * Hz.shape[-1])


if __name__ == '__main__':
    n = 100
    winds = np.linspace(10, 24, n)
    pool = multiprocessing.Pool(7)

    fig = plt.figure()
    ax = fig.add_subplot()

    for power in np.linspace(0, 15e6, 6):
        print(power)

        results = pool.starmap(change_power, (zip(winds, [power] * n)))

        z0s = np.hstack(results)
        p = ax.scatter(z0s[0, :] / DEG2RAD, z0s[2, :] / RPM2RAD, c=winds)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\Omega$')

    # ax.plot(z0s[0, :], z0s[2, :], 'r+', zdir='y', zs=0)
    # ax.plot(z0s[1, :], z0s[2, :], 'g+', zdir='x', zs=0)
    # ax.plot(z0s[0, :], z0s[1, :], 'k+', zdir='z', zs=0)

    fig.colorbar(p)

    fig2 = plt.figure()
    ax = fig2.add_subplot()

    for pitch in np.linspace(-4 * DEG2RAD, 20 * DEG2RAD, 6):
        print(pitch)

        results = pool.starmap(change_blade_pitch, (zip(winds, [pitch] * n)))

        z0s = np.hstack(results)
        p = ax.scatter(z0s[0, :] / DEG2RAD, z0s[2, :] / RPM2RAD, c=winds)

    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\Omega$')

    # ax.plot(z0s[0, :], z0s[2, :], 'r+', zdir='y', zs=0)
    # ax.plot(z0s[1, :], z0s[2, :], 'g+', zdir='x', zs=0)
    # ax.plot(z0s[0, :], z0s[1, :], 'k+', zdir='z', zs=0)

    fig2.colorbar(p)
    plt.show()
