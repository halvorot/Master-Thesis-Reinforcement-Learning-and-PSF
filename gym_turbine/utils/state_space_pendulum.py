import numpy as np
from numpy.linalg import inv

## Note: X = [q q_dot]^T in the state space X_dot = A*X+B*F_a

g = 9.81                            # Gravitational acceleration
DoFs = 3
zero_4_2 = np.zeros((4, 2))
zero_2_4 = np.zeros((2, 4))

# Turbine parameters
J = 1e8                             # Rotational intertia of pendulum [kg*m^2]
height = 100
m_a = 1e5                           # DVA mass
k_a = 6.18e4                        # DVA stiffness [N/m]
zeta_d = 0.6                        # DVA damping ratio
c_a = 2*zeta_d*np.sqrt(m_a*k_a)
spoke_length = 27                   # Horizontal distance from pendulum center to DVA
max_input = 1e6                     # DVA maximum force input [N]

T_W = 0                             # Wind force to pendulum pitch [N*m]

F_d = T_W                           # Disturbance forces

F_g = m_a*g

def M():
    M = np.array([  [J+2*m_a*spoke_length, m_a*spoke_length, m_a*spoke_length],
                    [m_a*spoke_length, m_a, 0],
                    [-m_a*spoke_length, 0, m_a]
    ])

    return M

def M_inv():
    return inv(M())

def C():

    C = np.diag([0, c_a, c_a])

    return C

def K():

    K = np.diag([0, k_a, k_a])

    return K

def B_Fa():

    B_Fa = np.vstack([[spoke_length, -spoke_length], -np.identity(2)])

    return B_Fa

def B_Fd():
    return np.array([1, 0, 0])

def B_Fg():
    return np.array([0, 1, 1])


def A():
    top = np.hstack((np.zeros((DoFs, DoFs)), np.identity(DoFs)))
    bottom = np.hstack((-M_inv().dot(K()), -M_inv().dot(C())))

    A = np.vstack((top, bottom))
    return A

def B():
    left = np.vstack((np.zeros((DoFs, DoFs)), M_inv()))

    B = left.dot(B_Fa())

    return B

def W():
    left = np.vstack((np.zeros((DoFs, DoFs)), M_inv()))

    W = left.dot(B_Fd())

    return W

def W_g():
    left = np.vstack((np.zeros((DoFs, DoFs)), M_inv()))

    W_g = left.dot(B_Fg())

    return W_g
