import numpy as np

g = 9.81                            # Gravitational acceleration
DoFs = 1

# Turbine parameters
L = 100
L_P = 30
m = 1e5
k = 1e6
zeta = 1
c = 2*zeta*np.sqrt(m*k)
J = (1/12)*m*(L**2 + 2*L*L_P + 6*L + L_P**2 - 6*L_P)
max_input = 5e6                     # maximum force input [N]