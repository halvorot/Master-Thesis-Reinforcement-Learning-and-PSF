import numpy as np

g = 9.81                            # Gravitational acceleration
DoFs = 3

# Turbine parameters
J = 1e8                             # Rotational intertia of pendulum [kg*m^2]
height = 100
l_BG = 30
m_a = 1e5                           # DVA mass
k_a = 6e4                           # DVA stiffness [N/m]
zeta_a = 0.6                        # DVA damping ratio
c_a = 2*zeta_a*np.sqrt(m_a*k_a)
spoke_length = 27                   # Horizontal distance from pendulum center to DVA
max_input = 1e6                     # DVA maximum force input [N]

T_W = 0                             # Wind force to pendulum angle [N*m]

F_d = T_W                           # Disturbance forces