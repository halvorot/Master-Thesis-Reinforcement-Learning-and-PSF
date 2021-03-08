import numpy as np

# Platform parameters
L = 144.45
L_thr = 50

# Rotor parameters
B = 0.97                                # Tip loss parameter (table 5.1 Pedersen)
lambda_star = 9                       # (table 5.1 Pedersen)
C_P_star = 0.489                         # (table 5.1 Pedersen)
C_F = 0.8                            # (table 5.1 Pedersen)
rho = 1.225                             # Air density (table 5.1 Pedersen)
R = 120                                  # Blade length
A = np.pi*R**2
R_p = B*R
A_p = np.pi*R_p**2
k_r = (2*rho*A_p*R)/(3*lambda_star)     # k in Force/Torque equations
l_r = (2/3)*R_p                         # l in Force/Torque equations
b_d_r = 0.5*rho*A*(B**2*(16/27)-C_P_star)*(R/lambda_star)**3  # b_d in Force/Torque equations
d_r = 0.5*rho*A*C_F                     # d in Force/Torque equations
J_r = 4.068903574982517e+07             # From OpenFast

max_thrust_force = 5e5                  # maximum force input [N]
blade_pitch_max = 20*(np.pi/180)        # maximum blade pitch angle (-+)
max_wind_force = 3e6                    # Just used for animation scaling
max_blade_pitch_rate = 8*(np.pi/180)    # Max blade pitch rate [rad/sec]
tau_blade_pitch = 0.1                   # Blade pitch time constant
tau_thr = 5                             # Thrust force time constant
omega_setpoint = 1.25*0.6