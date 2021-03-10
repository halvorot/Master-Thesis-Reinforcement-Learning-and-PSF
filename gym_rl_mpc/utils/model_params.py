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
tau_thr = 2                             # Thrust force time constant

def power_regime(wind_speed):
    if wind_speed < 3:
        return 0
    elif wind_speed < 10.59:
        return (((wind_speed-3)/(10.59-3))**2)*15e6
    else:
        return 15e6

def omega_setpoint(wind_speed):
    if wind_speed < 6.98:
        return 5*(2*np.pi/60)       # 5 rpm
    elif wind_speed < 10.59:
        return ((7.55-5)/(10.59-6.98))*(wind_speed-6.98)+5
    else:
        return 7.55*(2*np.pi/60)    # 7.55 rpm
