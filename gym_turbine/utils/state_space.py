import numpy as np
import gym_auv.utils.geomutils as geom
from numpy.linalg import inv
from math import cos, sin


I3 = np.identity(3)
zero3 = 0*I3
g = 9.81

# AUV parameters
m = 18                      # kg
W = m*g                     # N
_B = W+1                    # N
d = 0.15                    # meters
r = d/2                     # meters
L = 1.08                    # meters
z_G = 0.01                  # meters
r_G = [0, 0, z_G]           # meters
thrust_min = 0              # N
thrust_max = 14             # N
rudder_max = 30*np.pi/180   # rad
fins_max = 30*np.pi/180     # rad
U_max = 2                   # m/s

# Moments of inertia
I_x = (2/5)*m*r**2
I_y = (1/5)*m*((L/2)**2 + r**2)
I_z = I_y

Ig = np.vstack([
    np.hstack([I_x, 0, 0]),
    np.hstack([0, I_y, 0]),
    np.hstack([0, 0, I_z])])

# Added mass parameters (formulas from Fossen(2020))
e = 1 - (r/(L/2))**2
alfa_0 = 2*(1-e**2)/(e**3) * ((1/2)*np.log((1+e)/(1-e))-e)
beta_0 = 1/(e**2) - (1-(e**2))/(2*e**3)*np.log((1+e)/(1-e))
k1 = alfa_0 / (2-alfa_0)
k2 = beta_0 / (2-beta_0)
k = (e**4)*(beta_0-alfa_0)/((2-e**2)*(2*e**2 - (2-e**2)*(beta_0-alfa_0)))

X_udot = -m*k1
Y_vdot = -m*k2
Z_wdot = -m*k2
K_pdot = 0
M_qdot = -k*I_y
N_rdot = -k*I_z

# Linear damping parameters
X_u = -2.4
Y_v = -23
Y_r = 11.5
Z_w = Y_v
Z_q = -Y_r
K_p = -0.3
M_w = 3.1
M_q = -9.7
N_v = -M_w
N_r = M_q

# Nonlinear damping parameters
X_uu = -2.4
Y_vv = -80
Y_rr = 0.3
Z_ww = Y_vv
Z_qq = -Y_rr
K_pp = -6e-4
M_ww = 1.5
M_qq = -9.1
N_vv = -M_ww
N_rr = M_qq

# Lift parameters
C_LB = 1.24     # empirical body-lift coefficient
C_LF = 3        # empirical fin-lift coefficient
S_fin = 64e-4   # m^2
x_b = -0.4      # m
x_fin = -0.4    # m
rho = 1000      # kg/m^3

# Body Lift
Z_uwb = -0.5*rho*np.pi*(r**2)*C_LB
M_uwb = -(-0.65*L-x_b)*Z_uwb
Y_uvb = Z_uwb
N_uvb = -M_uwb

# Fin lift in Y and N
Y_uvf = rho*C_LF*S_fin*(-1)
N_uvf = x_fin*Y_uvf
Y_urf = rho*C_LF*S_fin*(-x_fin)
N_urf = x_fin*Y_urf
Y_uudr = rho*C_LF*S_fin
N_uudr = x_fin*Y_uudr

# Fin lift in Z and M
Z_uwf = -rho*C_LF*S_fin
M_uwf = -x_fin*Z_uwf
Z_uqf = -rho*C_LF*S_fin*(-x_fin)
M_uqf = -x_fin*Z_uqf
Z_uuds = -rho*C_LF*S_fin
M_uuds = -x_fin*Z_uuds

def M_RB():
    M_RB_CG = np.vstack([
        np.hstack([m*I3, zero3]),
        np.hstack([zero3, Ig])
    ])

    M_RB_CO = geom.move_to_CO(M_RB_CG, r_G)
    return M_RB_CO


def M_A():
    M_A = -np.diag([X_udot, Y_vdot, Z_wdot, K_pdot, M_qdot, N_rdot])
    return M_A


def M_inv():
    M = M_RB() + M_A()
    return inv(M)


def C_RB(nu):
    nu_2 = nu[3:6]

    Ib = Ig - m*geom.S_skew(r_G).dot(geom.S_skew(r_G))

    C_RB_CO = np.vstack([
        np.hstack([m*geom.S_skew(nu_2), -m*geom.S_skew(nu_2).dot(geom.S_skew(r_G))]),
        np.hstack([m*geom.S_skew(r_G).dot(geom.S_skew(nu_2)), -geom.S_skew(Ib.dot(nu_2))])
    ])
    return C_RB_CO


def C_A(nu):
    u = nu[0]
    v = nu[1]
    w = nu[2]
    p = nu[3]
    q = nu[4]
    r = nu[5]

    C_11 = np.zeros((3, 3))
    C_12 = np.array([[0, -Z_wdot*w, Y_vdot*v],
                     [Z_wdot*w, 0, -X_udot*u],
                     [-Y_vdot*v, X_udot*u, 0]])

    C_21 = np.array([[0, -Z_wdot*w, Y_vdot*v],
                     [Z_wdot*w, 0, -X_udot*u],
                     [-Y_vdot*v, X_udot*u, 0]])

    C_22 = np.array([[0, -N_rdot*r, M_qdot*q],
                     [N_rdot*r, 0, -K_pdot*p],
                     [-M_qdot*q, K_pdot*p, 0]])

    C_A = np.vstack([np.hstack([C_11, C_12]), np.hstack([C_21, C_22])])
    return C_A


def C(nu):
    C = C_RB(nu) + C_A(nu)
    return C


def D(nu):
    u = abs(nu[0])
    v = abs(nu[1])
    w = abs(nu[2])
    p = abs(nu[3])
    q = abs(nu[4])
    r = abs(nu[5])

    D = -np.array([[X_u, 0, 0, 0, 0, 0],
                   [0, Y_v, 0, 0, 0, Y_r],
                   [0, 0, Z_w, 0, Z_q, 0],
                   [0, 0, 0, K_p, 0, 0],
                   [0, 0, M_w, 0, M_q, 0],
                   [0, N_v, 0, 0, 0, N_r]])
    D_n = -np.array([[X_uu*u, 0, 0, 0, 0, 0],
                     [0, Y_vv*v, 0, 0, 0, Y_rr*r],
                     [0, 0, Z_ww*w, 0, Z_qq*q, 0],
                     [0, 0, 0, K_pp*p, 0, 0],
                     [0, 0, M_ww*w, 0, M_qq*q, 0],
                     [0, N_vv*v, 0, 0, 0, N_rr*r]])
    L = -np.array([[0, 0, 0, 0, 0, 0],
                   [0, Y_uvb+Y_uvf, 0, 0, 0, Y_urf],
                   [0, 0, Z_uwb+Z_uwf, 0, Z_uqf, 0],
                   [0, 0, 0, 0, 0, 0],
                   [0, 0, M_uwb+M_uwf, 0, M_uqf, 0],
                   [0, N_uvb+N_uvf, 0, 0, 0, N_urf]])
    return D + D_n + L*u


def B(nu):
    u = nu[0]
    B = np.array([[1, 0, 0],
                  [0, Y_uudr*(u**2), 0],
                  [0, 0, Z_uuds*(u**2)],
                  [0, 0, 0],
                  [0, 0, M_uuds*(u**2)],
                  [0, N_uudr*(u**2), 0]])
    return B


def G(eta):
    phi = eta[3]
    theta = eta[4]
    G = np.array([(W-_B)*sin(theta),
                  -(W-_B)*cos(theta)*sin(phi),
                  -(W-_B)*cos(theta)*cos(phi),
                  z_G*W*cos(theta)*sin(phi),
                  z_G*W*sin(theta),
                  0])
    return G
