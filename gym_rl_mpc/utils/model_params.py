import numpy as np

g = 9.81                            # Gravitational acceleration
DoFs = 1

# Turbine parameters
L = 100
m = 1e5
k = 1e6
zeta = 1
c = 2*zeta*np.sqrt(m*k)
J = (1/3)*m*L**2
max_input = 3e6                     # maximum force input [N]

T_W = 0                             # Wind force to pendulum angle [N*m]

F_d = T_W                           # Disturbance forces