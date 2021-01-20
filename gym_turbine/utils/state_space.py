import numpy as np
from numpy.linalg import inv

## Note: X = [q q_dot]^T in the state space X_dot = A*X+B*F_a

### QUESTIONS ###
# Are C_d (eq 15) and C (in eq 22) the same matrix?
# Is M_d (in eq 2) supposed to be m_d?
# Is k_d = 1.91e5 or 6.18e4
# How do we control wind speed and disturbances? Have to calculate F_d somehow?

DoFs = 11
zero_4_2 = np.zeros((4, 2))
zero_2_4 = np.zeros((2, 4))

# Turbine parameters
m_s = 2.38e7                        # Platform mass for surge and sway motion [kg]
m_hv = 1.08e7                       # Platform mass for heave motion [kg]
J_p = 8.15e9                        # Rotational intertia of platform [kg*m^2]
l_c = 60                            # TODO: made up number!! change!! Vertical distance from mean sea level to platform mass center [m]
m_d = 1e5                           # DVA mass [kg]
k_d = 6.18e4                        # TODO: CHECK, DVA stiffness [N/m]
zeta_d = 0.6                        # DVA damping ratio
c_d = 2*zeta_d*np.sqrt(m_d*k_d)     # TODO: CHECK, DVA damping coefficient
m_tf = 3.5e5                        # TODO: made up number!! change!!
m_ts = 3.5e5                        # TODO: made up number!! change!!
c_tf = 7.70e4                       # Damping coefficient of tower top fore‐aft motion [N/(m/s)]
c_ts = 6.67e3                       # Damping coefficient of tower top side-side motion [N/(m/s)]
c_s = 4.861e5                       # Damping coefficient of platform surge and sway [N/(m/s)]
c_t = 13263                         # Structural damping coefficient of tower [N/(m/s)]
H = 129.962                         # Distance from nacelle center to platform mass center [m]
c_ph = 1.60e4                       # Damping coefficient of platform pitch and roll [N/(m/s)]
c_hv = 1e4                          # TODO: made up number!! change!!
k_s = 3.374e6                       # Spring stiffness of platform surge and sway [N/m]
k_t = 2.41e6                        # Stiffness of tower [N/m]
k_B = 2.559e6                       # Buoyant stiffness [N/m]
k_ph = 2.581e7                      # Stiffness of mooring lines [N/m]
k_sh = -0.0124                      # Couple parameter between surge and heave
l = 27                              # Platform spoke length [m]
l_s = 47.89                         # Vertical distance from mean sea level to platform bottom

dva_max = 5e9                      # TODO: made up number!! change!! DVA maximum force input [N]

F_Wsg = 0                           # Wave force to platform surge [N]
F_Wsw = 0                           # Wave force to platform sway [N]
F_Whv = 0                           # Wave force to platform heave [N]
T_Wr = 0                            # Wave force to platform roll [N*m]
T_Wp = 0                            # Wave force to platform pitch [N*m]
F_thr = 0                           # Thrust force [N]
F_ts = 0                            # Force in side-side direction [N]

F_d = [F_Wsg, F_Wsw, F_Whv, T_Wr, T_Wp, F_thr, F_ts]     # Disturbance forces

def M():
    M = np.diag([m_s, m_s, m_hv, J_p + m_s*l_c**2, J_p + m_s*l_c**2, m_tf, m_ts, m_d, m_d, m_d, m_d])
    M[4, 0] = -m_s*l_c
    M[3, 1] = m_s*l_c
    M[1, 3] = m_s*l_c
    M[0, 4] = -m_s*l_c

    return M

def M_inv():
    return inv(M())

def C_d(gamma):

    C_upper = np.array([[c_s + c_t, 0,          0,                  0,                          H*c_t,                      -c_t,   0,      0,                      0,                      0,                      0],
                        [0,         c_s + c_t,  0,                  H*c_t,                      0,                          0,      -c_t,   0,                      0,                      0,                      0],
                        [0,         0,          4*(c_ph+c_d)+c_hv,  0,                          0,                          0,      0,      -c_d,                   -c_d,                   -c_d,                   -c_d],
                        [0,         H*c_t,      0,                  2*(c_ph+c_d)*l**2+H**2*c_t, 0,                          0,      -c_t*H, -c_d*l*np.sin(gamma),   -c_d*l*np.cos(gamma),   c_d*l*np.sin(gamma),   c_d*l*np.cos(gamma)],
                        [H*c_t,     0,          0,                  0,                          2*(c_ph+c_d)*l**2+H**2*c_t, -c_t*H, 0,      c_d*l*np.cos(gamma),    -c_d*l*np.sin(gamma),   -c_d*l*np.cos(gamma),   c_d*l*np.sin(gamma)]])

    ct_diag = np.diag([-c_t, -c_t])
    cd_stack = np.vstack([-c_d, -c_d, -c_d, -c_d])

    lower_left = np.vstack((np.hstack((ct_diag, np.vstack((0, 0)))), np.hstack((zero_4_2, cd_stack))))
    lower_mid = np.array([  [0,                     -c_t*H],
                            [-c_t*H,                0],
                            [-c_d*l*np.sin(gamma),  c_d*l*np.cos(gamma)],
                            [-c_d*l*np.cos(gamma),  -c_d*l*np.sin(gamma)],
                            [c_d*l*np.sin(gamma),   -c_d*l*np.cos(gamma)],
                            [c_d*l*np.cos(gamma),   c_d*l*np.sin(gamma)]])

    lower_right = np.diag([c_tf+c_t, c_ts+c_t, c_d, c_d, c_d, c_d])

    C_lower = np.hstack((lower_left, lower_mid, lower_right))

    C_d = np.vstack((C_upper, C_lower))

    return C_d

def K(gamma):

    K_upper = np.array([[k_s + k_t,                 0,              0,                  0,                                      k_t*H - k_s*l_s,                        -k_t,   0,      0,                      0,                      0,                      0],
                        [0,                         k_s + k_t,      0,                  k_t*H + k_s*l_s,                        0,                                      0,      k_t,    0,                      0,                      0,                      0],
                        [k_sh*(4*(k_ph+k_d)+k_B),   0,              4*(k_ph+k_d)+k_B,   0,                                      0,                                      0,      0,      -k_d,                   -k_d,                   -k_d,                   -k_d],
                        [0,                         k_t*H+k_s*l_s,  0,                  2*(k_ph+k_d)*l**2+k_t*H**2+k_s*l_s**2,  0,                                      0,      k_t*H,  -k_d*l*np.sin(gamma),   -k_d*l*np.cos(gamma),   k_d*l*np.sin(gamma),   k_d*l*np.cos(gamma)],
                        [k_t*H-k_s*l_s,             0,              0,                  0,                                      2*(k_ph+k_d)*l**2+k_t*H**2+k_s*l_s**2,  -k_t*H, 0,      k_d*l*np.cos(gamma),   -k_d*l*np.sin(gamma),   -k_d*l*np.cos(gamma),   k_d*l*np.sin(gamma)]])

    kt_skew = np.diag([-k_t, k_t])
    kd_stack = np.vstack([-k_d, -k_d, -k_d, -k_d])

    lower_left = np.vstack((np.hstack((kt_skew, np.vstack((0, 0)))), np.hstack((zero_4_2, kd_stack))))
    lower_mid = np.array([  [0,                     -k_t*H],
                            [k_t*H,                 0],
                            [-k_d*l*np.sin(gamma),  k_d*l*np.cos(gamma)],
                            [-k_d*l*np.cos(gamma),  -k_d*l*np.sin(gamma)],
                            [k_d*l*np.sin(gamma),   -k_d*l*np.cos(gamma)],
                            [k_d*l*np.cos(gamma),   k_d*l*np.sin(gamma)]])

    lower_right = np.diag([k_t, k_t, k_d, k_d, k_d, k_d])

    K_lower = np.hstack((lower_left, lower_mid, lower_right))

    K = np.vstack((K_upper, K_lower))

    return K

def B_Fa(gamma):
    mid = np.array([[l*np.sin(gamma),   l*np.cos(gamma), -l*np.sin(gamma), -l*np.cos(gamma)],
                    [-l*np.cos(gamma),  l*np.sin(gamma), l*np.cos(gamma),  -l*np.sin(gamma)]])
    B_Fa = np.vstack([zero_2_4, [1, 1, 1, 1], mid, zero_2_4, -np.identity(4)])

    return B_Fa

def B_Fd():
    return np.vstack((np.identity(7), np.zeros((4, 7))))


def A(gamma):
    top = np.hstack((np.zeros((DoFs, DoFs)), np.identity(DoFs)))
    bottom = np.hstack((-M_inv().dot(K(gamma)), -M_inv().dot(C_d(gamma))))

    A = np.vstack((top, bottom))
    return A

def B(gamma):
    left = np.vstack((np.zeros((DoFs, DoFs)), M_inv()))

    B = left.dot(B_Fa(gamma))

    return B

def W():
    left = np.vstack((np.zeros((DoFs, DoFs)), M_inv()))

    W = left.dot(B_Fd())

    return W
