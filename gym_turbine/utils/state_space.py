import numpy as np
from numpy.linalg import inv


I3 = np.identity(3)
zero3 = 0*I3
g = 9.81

# Turbine parameters
water_depth = 200       # m
rotor_diameter = 126    # m
hub_height = 90         # m
mass = 8.6e6            # kg

# Moments of inertia
I_x = 1
I_y = 1
I_z = I_y

Ig = np.vstack([
    np.hstack([I_x, 0, 0]),
    np.hstack([0, I_y, 0]),
    np.hstack([0, 0, I_z])])


def M_RB():
    M_RB_CG = np.vstack([
        np.hstack([mass*I3, zero3]),
        np.hstack([zero3, Ig])
    ])

    M_RB_CO = M_RB_CG
    return M_RB_CO

def M_inv():
    M = M_RB()
    return inv(M)
