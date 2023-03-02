
import numpy as np
import CEMD_1pUC as cemd
import matplotlib.pyplot as plt

# Define properties and parameters

a = 300e-9                   # Lattice constant along the "x" axis
b = 300e-9                   # Lattice constant along the "y" axis

n_p = 3.5                    # Particle refractive index divided by vacuum refractive index
R_p = a/4                    # Particle radius

norm_freq = np.linspace(0.3,0.9,101)         # Range of normalized frequency = ka/(2pi)
theta = np.linspace(0,89.99,101)*np.pi/180    # Range of angle of incidence

k = 2*np.pi/a*norm_freq                      # Medium wavevector


# Reflectance and transmittance calculation

R_TM, R_TE, T_TM, T_TE = cemd.get_RT(a,b,R_p,n_p,k,theta)

# Save data

savefile = 'RTM_SquareArray_ab300nm_n35_Ra4_freq03-09_theta0-89.dat'
    
with open(savefile, 'wb') as f:
  np.save(f, R_TM)

savefile = 'RTE_SquareArray_ab300nm_n35_Ra4_freq03-09_theta0-89.dat'
    
with open(savefile, 'wb') as f:
  np.save(f, R_TE)
  
  
