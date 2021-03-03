import numpy as np

g = 9.81                            # Gravitational acceleration
DoFs = 1

# Turbine parameters
L = 144.495
L_P = 50
L_COM = 6                          # Meters below rotation point
m = 1.7838E+07 + 190000 + 507275   # Platform mass + Hub mass + Nacelle mass 
k = 4.5e4
zeta = 1
c = 2*zeta*np.sqrt(m*k)
J = 2*1.2507E+10 + m*(L_COM)**2     # J_COM + m*d^2
max_input = 1e6                     # maximum force input [N]
omega_disturbance = 1
max_disturbance = 1e7