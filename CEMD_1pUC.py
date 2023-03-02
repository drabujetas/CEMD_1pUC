# Author: Diego Romero Abujetas, March 2023, diegoromeabu@gmail.com
#
# This file contain all the functions needed to calculate reflectance and transmittance.
# of a free standing metasurface of one particle per unit cell. The list of functions included in this file are:
#
# Calculation of the depolarization Green function for 1 particle per unit cell .
# - GbCalc_1UC_Cyx_th_mode, Gb_Ch, Gb1D_kx 		
# Calculation of reflectance and transmittance.
# - calc_rt,  calc_RT, calc_RT_pol
# Calculation of (dipolar) polarizability of spheres by Mie theory.
# - psi, diff_psi, xi, diff_xi 			(Auxiliary functions)
# - Mie_an, Mie_bn				(Mie coefficients)
# - get_alpha					(dipolar polarizability)
# Full calculation of reflectance and transmittance.
# - get_RT, get rt
#
# The next libraries are used

import numpy as np
import scipy.special as sps
from mpmath import mp

# GbCalc_1UC: Function that calculates the Green function ("GbCalc") 
# for an 2D array with one particle per unit cell ("1UC").
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while for "th = pi/3" and "a = b" a triangular lattice is recovered.
#
# "k" is the wavector in the metasurface medium ()"k = k0*n_bg").
#
# "kx" and "ky" are the projection of the wavevector along "x" and "y" axis.
# They are real quantities, while "k" can be complex.
#
# "N" and "Ni" are the number of elements taken in the sum. For bigger "N"
# the convergence is better, but be careful with "Ni". With "Ni = 4" is
# enough to get a good convergence, but for bigger "Ni" is necessary to take
# bigger "N" to get convergence. See PRB 2020, D. R. Abuejetas et. al., 102, 125411 
# for more information about the convergence and the menaing of "Ni". 
#
# Output:
#
# "Gga" is a 6 by 6 matrix with the value of the depolarization Green function.

def GbCalc_1UC(a, b, k, kx, ky, th = np.pi/2, N = 100):

    kx = kx - np.floor( (kx + np.pi/a)/(2*np.pi/a))*(2*np.pi/a)   # bring "kx" to the first Brilluoin zone
    ky = ky - np.floor( (ky + np.pi/b)/(2*np.pi/b))*(2*np.pi/b)

    Ni = int(np.floor( np.real(k + np.abs(kx))/(2*np.pi/a) ) + 3)

    GbCh = Gb_Ch(a,k,kx)
    Gb1D = Gb1D_kx(N,b*np.sin(th),k,ky,kx)

    for m in range(Ni-1):

        kxlp = kx - 2*np.pi/a*(m + 1)
        kxlm = kx + 2*np.pi/a*(m + 1)
        Nm = N*(m+1)
        Gb1D = Gb1D + Gb1D_kx(Nm,b*np.sin(th),k,ky - ((kxlp-kx)*np.cos(th)/np.sin(th)),kxlp) + Gb1D_kx(Nm,b*np.sin(th),k,ky - ((kxlm-kx)*np.cos(th)/np.sin(th)),kxlm)

    Gb2D = GbCh + 1/a*(Gb1D)
    Gga = Gb2D

    return Gga


# Gb1D_kx: Function for calculating the depolarization Green function for an array of 1D cylinders
# or particles with translational symmetry along the "x" axis ("Gb1D") where the projection of the 
# wavevector can be also along the "x" axis ("kx"), where the convergence of the sums are below "to 1/m^3" (s3). 
# Therefore, the cylinder axis is along the "x"  axis and they are periodically spaced along the "y" axis. 
# The sum is done in reciprocal space.
# 
# This function do the same that "Gb1D_kx", but here is rewritten to avoid recalculation, and 
# also all the sums are expressed with convergence better than "1/m^3".
#
# Inputs:
#
# "N" is the number of terms taken in the sums.
#
# "b" is the distance between the particles (along the "y" axis).
#
# "k" is the wavector in the metasurface medium ()"k = k0*n_bg").
#
# "kx" and "ky" are the projection of the wavevector along "x" and "y" axis.
# "kx" is along the translational symmetry axis and "ky" along the direction of the periodicity.
# They are real quantities, while "k" can be complex.
#
# Output:
#
# "Gb1D" is a 6 by 6 matrix with the depolarization Green tensor.
    
def Gb1D_kx(N,b,k,ky,kx):

    gE = 0.577215664901532860606512090082402431042
    zR_3 = 1.2020569031595942853997381615114499907 #zeta Riemann evaluated at z = 3 
    f3 = (b/(2*np.pi))**3*zR_3
    
    m = np.linspace(1,N,N)
             
    kp = np.sqrt(k ** 2 - kx ** 2, dtype = 'complex_')
    kz = np.sqrt(kp ** 2 - ky ** 2, dtype = 'complex_')
    
    km = 2*np.pi*m/b
    kym = ky - km 
    kymm = ky + km
    kzm = np.sqrt(kp ** 2 - kym ** 2, dtype = 'complex_')
    kzmm = np.sqrt(kp ** 2 - kymm ** 2, dtype = 'complex_')
    
    fxx3 = (kz**2 + 3*ky**2)
    fyy3 = (4*k**2*kz**2 + 12*k**2*ky**2 - 10*ky**2*kz**2 - 7*ky**4 - 3*kz**4)/4
    fzz3 = (4*k**2*kz**2 + 12*k**2*ky**2 - 6*ky**2*kz**2 - 5*ky**4 - kz**4)/4
    fxy3 = 2j*kp**2*ky
    
    Sum1 = (1j*(1/(2*kz*b) - 1./4) + 1/(2*b)*(np.sum(1j/kzm + 1j/(kzmm) - 2/km - fxx3/km**3) + fxx3*f3) + 1/(2*np.pi) * (np.log(kp*b/(4*np.pi)) + gE))
    Sum2 = -(1/k*(1j*ky/(2*kz*b) + 1j/(2*b)*(np.sum(kym/kzm + kymm/kzmm - fxy3/km**3) + fxy3*f3) - 1/(2*np.pi)*ky))
    
    Gbxx = kp ** 2/k ** 2 * Sum1    
    Gbyy = (1j/(2*kz*b)*(1 - ky ** 2/k ** 2) - 1j/8*(1 + kx ** 2/k ** 2) + 1/(2*k ** 2*b)*(np.sum(1j*(k ** 2 - kym ** 2)/kzm 
         + 1j * (k ** 2 - kymm ** 2)/kzmm - 1/km*(k ** 2 + kx ** 2 - 2*km ** 2) - fyy3/km**3 ) + fyy3*f3 ) + 1/(4*np.pi*k ** 2)*(np.log(kp*b/(4*np.pi)) + gE )*(k ** 2 + kx ** 2)
         + 1/(8*np.pi*k ** 2)*(ky ** 2 - kz ** 2) + 1/6*np.pi/(k ** 2*b ** 2) )
    Gbzz = ( 1j/(2*kz*b)*(1 - kz ** 2/k ** 2) - 1j/8*(1 + kx ** 2/k ** 2) + 1/(2*k ** 2*b)*(np.sum(1j*(k ** 2 - kzm ** 2)/kzm 
         + 1j*(k ** 2 - kzmm ** 2)/kzmm - 1/km*(k ** 2 + kx ** 2 + 2*km ** 2) - fzz3/km**3 ) + fzz3*f3 ) + 1/(4*np.pi*k ** 2)*(np.log(kp*b/(4*np.pi)) + gE )*(k ** 2 + kx ** 2)
         + 1/(8*np.pi*k ** 2)*(kz ** 2 - ky ** 2) - 1/6*np.pi/(k ** 2*b ** 2) )
    Gbxy = kx/k * Sum2
    Gbyz = (kx/k * Sum1)
    Gbzx = -Sum2
    
    Gb1D = np.array([[Gbxx, Gbxy, 0,0,0,-Gbzx],[Gbxy, Gbyy, 0, 0,0,Gbyz], [0,0,Gbzz, Gbzx,-Gbyz,0], [0,0,Gbzx,Gbxx,Gbxy,0] , [0,0,-Gbyz,Gbxy,Gbyy,0], [-Gbzx,Gbyz,0,0,0,Gbzz]]) 
    
    return Gb1D


# "Gb_Ch" calculates the depolarization Green function of chain of particles align along the "x" axis
# oriented for the calculation of "Gb" of an two dimensional array.
#
# Inputs:
#
# "d" is the distance between particles.
# "k" is the wavevector in the medium (It can be complex).
# "kp" is the projection of the wavevector over the axis of the chain (the "x" axis) (It is real).
#
# Outputs:
#
# GbCh is the contribution of the chain to the depolarization Green function of the two dimensional array.

def Gb_Ch(d,k,kp):

    L1m = - np.log(1 - np.exp(1j * (k-kp) * d))#mp.polylog(1,np.exp(1j * (k-kp) * d))
    L1p = - np.log(1 - np.exp(1j * (k+kp) * d))#mp.polylog(1,np.exp(1j * (k+kp) * d))
    L2m = mp.polylog(2,np.exp(1j * (k-kp) * d))
    L2p = mp.polylog(2,np.exp(1j * (k+kp) * d))
    L3m = mp.polylog(3,np.exp(1j * (k-kp) * d))
    L3p = mp.polylog(3,np.exp(1j * (k+kp) * d))
    
    fac = 1j/(4*d ** 3 * k ** 2 * np.pi)
    dk = d * k
    dk2 = d ** 2 * k ** 2
    
    Gbxx = -2*fac*(dk * (L2m + L2p ) + 1j*( L3m + L3p ))
    Gbyy = fac * ( - 1j * dk2 *( L1m + L1p ) + dk * ( L2m + L2p ) + 1j *( L3m +  L3p ))
    Gbyz = fac*( - 1j * dk2 * ( L1m - L1p ) + dk *( L2m - L2p ))
          
    Gb_Ch = np.zeros((6, 6), dtype = 'complex_' )

    Gb_Ch[0,0] = Gbxx
    Gb_Ch[1,1] = Gbyy
    Gb_Ch[2,2] = Gbyy
    Gb_Ch[3,3] = Gbxx
    Gb_Ch[4,4] = Gbyy
    Gb_Ch[5,5] = Gbyy

    Gb_Ch[1,5] = Gbyz
    Gb_Ch[2,4] = - Gbyz
    Gb_Ch[4,2] = - Gbyz
    Gb_Ch[5,1] = Gbyz
    
    return Gb_Ch

    
# calc_rt: Function that calculates complex the specular reflection and transmission (rt) for TE and TM incidence.
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "Gb" is the depolarization Green function, a "6 by 6" matrix.
#
# "alp" is a "6 by 6" matrix with the electric and magnetic polarizability of the particles.
#
# "k" is a scalar with the values of the wavevector in the external media ("k = k0*n_b").
#
# "kx" and "ky" are the projection of the wavevector along "x" and "y" axis.
#
# Outputs:
#
# The output are the complex reflection and transmission (scalars) for different incident
# polarizations (rTM, tTM, rTE, tTE).
#

def calc_rt(a, b, Gb, alp, k, kx, ky, th = np.pi/2):
    
    alp = k ** 2 * alp   
    Gbalp = np.linalg.inv( np.eye(6) - np.dot(Gb, alp) )

    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2)
    ang1 = np.arccos(kz/k)
    alpha2 = np.arctan2(ky,kx)
        
    Gfeer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,kz*ky/k ** 2],[kx*kz/k ** 2,ky*kz/k ** 2,1-kz ** 2/k ** 2]])
    Gfmer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,kz/k,ky/k],[-kz/k,0,-kx/k],[-ky/k,kx/k,0]])
    Grf = np.zeros((6,6), dtype = 'complex_')
    Grf[0:3,0:3] = Gfeer
    Grf[0:3,3:6] = -Gfmer
    Grf[3:6,0:3] = Gfmer
    Grf[3:6,3:6] = Gfeer

    Gfeet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,-kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,-kz*ky/k ** 2],[-kx*kz/k ** 2,-ky*kz/k ** 2,1-kz ** 2/k ** 2]])
    Gfmet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,-kz/k,ky/k],[kz/k,0,-kx/k],[-ky/k,kx/k,0]])        
    Gtf = np.zeros((6,6), dtype = 'complex_')
    Gtf[0:3,0:3] = Gfeet
    Gtf[0:3,3:6] = -Gfmet
    Gtf[3:6,0:3] = Gfmet
    Gtf[3:6,3:6] = Gfeet

    EiTM = np.transpose(np.array([[np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1),-np.sin(alpha2),np.cos(alpha2),0]]))
    EiTE = np.transpose(np.array([[np.sin(alpha2),-np.cos(alpha2),0,np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1)]]))

    EfTM = np.dot(Gbalp, EiTM)
    EfTE = np.dot(Gbalp, EiTE)

    ErTM = Grf @ alp @ EfTM
    ErTE = Grf @ alp @ EfTE
    EtTM = (Gtf @ alp @ EfTM) + EiTM
    EtTE = (Gtf @ alp @ EfTE) + EiTE

    if np.abs(EiTM[4,0]) > np.abs(EiTM[3,0]):
        rTM = ErTM[4,0]/EiTM[4,0] 
        tTM = EtTM[4,0]/EiTM[4,0]
        rTE = ErTE[1,0]/EiTE[1,0]
        tTE = EtTE[1,0]/EiTE[1,0]
    else:
        rTM = ErTM[3,0]/EiTM[3,0] 
        tTM = EtTM[3,0]/EiTM[3,0]
        rTE = ErTE[0,0]/EiTE[0,0]
        tTE = EtTE[0,0]/EiTE[0,0]

    return rTM, rTE, tTM, tTE 
    
    
# calc_RT: Function that calculates the reflectance and transmittance (RT) for TE and TM incidence.
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "Gb" is the depolarization Green function, a "6 by 6" matrix.
#
# "alp" is a "6 by 6" matrix with the electric and magnetic polarizability of the particles.
#
# "k" is a scalar with the values of the wavevector in the external media ("k = k0*n_b").
#
# "kx0" and "ky0" are the proyection of the wavevector along "x" and "y" axis.
#
# "n" and "m" is the calculated diffraction order ("n = m = 0" is the specular mode).
#
# Outputs:
#
# The output are the reflectance and transmittance (scalars) for different incident
# polarizations (RTM, TTM, RTE, TTE).

def calc_RT(a, b, Gb, alp, k, kx0, ky0, th = np.pi/2, n = 0, m = 0):
    
    alp = k ** 2 * alp
    
    Gbalp = np.linalg.inv( np.eye(6) - np.dot(Gb, alp) )
    
    kz0 = np.sqrt(k ** 2 - kx0 ** 2 - ky0 ** 2) #by definition, must be real
    ang1 = np.arccos(kz0/k)
    alpha2 = np.arctan2(ky0,kx0)
    
    kx = kx0 - 2*np.pi/a*n
    ky = ky0 + 2*np.pi*n/a*np.sin(np.pi/2 - th)/np.cos(np.pi/2 - th) - 2*np.pi*m/(b*np.cos(np.pi/2 - th))
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')
    
    if np.imag(kz) == 0:
    
        Gfeer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,kz*ky/k ** 2],[kx*kz/k ** 2,ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        Gfmer = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,kz/k,ky/k],[-kz/k,0,-kx/k],[-ky/k,kx/k,0]])
        Grf = np.zeros((6,6), dtype = 'complex_')
        Grf[0:3,0:3] = Gfeer
        Grf[0:3,3:6] = -Gfmer
        Grf[3:6,0:3] = Gfmer
        Grf[3:6,3:6] = Gfeer

        Gfeet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,-kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,-kz*ky/k ** 2],[-kx*kz/k ** 2,-ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        Gfmet = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)*np.array([[0,-kz/k,ky/k],[kz/k,0,-kx/k],[-ky/k,kx/k,0]])        
        Gtf = np.zeros((6,6), dtype = 'complex_')
        Gtf[0:3,0:3] = Gfeet
        Gtf[0:3,3:6] = -Gfmet
        Gtf[3:6,0:3] = Gfmet
        Gtf[3:6,3:6] = Gfeet
        
        Sz = 1/(4)*np.array([[0,0,0,0,1,0],[0,0,0,-1,0,0],[0,0,0,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]])
        
        EiTM = np.transpose(np.array([[np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1),-np.sin(alpha2),np.cos(alpha2),0]]))
        EiTE = np.transpose(np.array([[np.sin(alpha2),-np.cos(alpha2),0,np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1)]]))
        
        if m == 0 and n == 0:
            w_t = 1
        else:
            w_t = 0
        
        EfTM = np.dot(Gbalp, EiTM)
        EfTE = np.dot(Gbalp, EiTE)
        
        ErTM = Grf @ alp @ EfTM
        EtTM = Gtf @ alp @ EfTM + EiTM*w_t
        ErTE = Grf @ alp @ EfTE
        EtTE = Gtf @ alp @ EfTE + EiTE*w_t
        
        EEiTM = np.transpose(np.conj(EiTM)) @ Sz @ EiTM 
        EEiTE = np.transpose(np.conj(EiTE)) @ Sz @ EiTE 

        RTM = - (np.transpose(np.conj(ErTM)) @ Sz @ ErTM)/EEiTM
        TTM = + (np.transpose(np.conj(EtTM)) @ Sz @ EtTM)/EEiTM
        RTE = - (np.transpose(np.conj(ErTE)) @ Sz @ ErTE)/EEiTE
        TTE = + (np.transpose(np.conj(EtTE)) @ Sz @ EtTE)/EEiTE
        
    else:
        
        TTM = 0 
        RTM = 0 
        TTE = 0
        RTE = 0 

    return np.real(RTM), np.real(RTE), np.real(TTM), np.real(TTE) 
    
    
# calc_RT: Function that calculates the reflectance and transmittance (RT) for an arbitrary plane wave incidence.
#
# Inputs:
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "Gb" is the depolarization Green function, a "6 by 6" matrix.
#
# "alp" is a "6 by 6" matrix with the electric and magnetic polarizability of the particles.
#
# "k" is a scalar with the values of the wavevector in the external media ("k = k0*n_b").
#
# "kx0" and "ky0" are the projection of the wavevector along "x" and "y" axis.
#
# "polTM" and "polTE" determine the projection of the incoming wave into "TM" and "TE" incidence.
# For exaple:
#	- "polTM = 1", "polTE = 0" --> TM incidence
#	- "polTM = 1", "polTE = 1j" --> circular polarized light
#
# "n" and "m" is the calculated diffraction order ("n = m = 0" is the specular mode).
#
# Outputs:
#
# The output are the reflectance and transmittance (scalars) at the specific polarization.
 
def calc_RT_pol(a, b, Gb, alp, k, kx0, ky0, polTM = 1, polTE = 1j, th = np.pi/2, n = 0, m = 0):
    
    alp = k ** 2 * alp
    
    Gbalp = np.linalg.inv( np.eye(6) - np.dot(Gb, alp) )
    
    kz0 = np.sqrt(k ** 2 - kx0 ** 2 - ky0 ** 2) #by definition, must be real
    ang1 = np.arccos(kz0/k)
    alpha2 = np.arctan2(ky0,kx0)
    
    kx = kx0 - 2*np.pi/a*n
    ky = ky0 + 2*np.pi*n/a*np.sin(np.pi/2 - th)/np.cos(np.pi/2 - th) - 2*np.pi*m/(b*np.cos(np.pi/2 - th))
    kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2, dtype = 'complex_')
    
    if np.imag(kz) == 0:
    
        pre_fac = 1j/(2*a*b*np.cos(np.pi/2 - th)*kz)
    
        Gfeer = pre_fac*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,kz*ky/k ** 2],[kx*kz/k ** 2,ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        Gfmer = pre_fac*np.array([[0,kz/k,ky/k],[-kz/k,0,-kx/k],[-ky/k,kx/k,0]])
        Grf = np.zeros((6,6), dtype = 'complex_')
        Grf[0:3,0:3] = Gfeer
        Grf[0:3,3:6] = -Gfmer
        Grf[3:6,0:3] = Gfmer
        Grf[3:6,3:6] = Gfeer

        Gfeet = pre_fac*np.array([[1-kx ** 2/k ** 2,-ky*kx/k ** 2,-kz*kx/k ** 2],[-kx*ky/k ** 2,1-ky ** 2/k ** 2,-kz*ky/k ** 2],[-kx*kz/k ** 2,-ky*kz/k ** 2,1-kz ** 2/k ** 2]])
        Gfmet = pre_fac*np.array([[0,-kz/k,ky/k],[kz/k,0,-kx/k],[-ky/k,kx/k,0]])        
        Gtf = np.zeros((6,6), dtype = 'complex_')
        Gtf[0:3,0:3] = Gfeet
        Gtf[0:3,3:6] = -Gfmet
        Gtf[3:6,0:3] = Gfmet
        Gtf[3:6,3:6] = Gfeet
            
        Sz = 1/(4)*np.array([[0,0,0,0,1,0],[0,0,0,-1,0,0],[0,0,0,0,0,0],[0,-1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0]])
              
        EiTM = np.transpose(np.array([[np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1),-np.sin(alpha2),np.cos(alpha2),0]]))
        EiTE = np.transpose(np.array([[np.sin(alpha2),-np.cos(alpha2),0,np.cos(ang1)*np.cos(alpha2),np.cos(ang1)*np.sin(alpha2),-np.sin(ang1)]]))
        Ei = (EiTM*polTM + EiTE*polTE)/np.abs(polTM + polTE)
        
        if m == 0 and n == 0:
            w_t = 1
        else:
            w_t = 0
        
        Ef = np.dot(Gbalp, Ei)
        
        Er = Grf @ alp @ Ef
        Et = Gtf @ alp @ Ef + Ei*w_t
        
        EEi = np.transpose(np.conj(Ei)) @ Sz @ Ei 

        R = - (np.transpose(np.conj(Er)) @ Sz @ Er)/EEi
        T = + (np.transpose(np.conj(Et)) @ Sz @ Et)/EEi
	                
    else:
        
        R = 0 
        T = 0 

    return np.real(R), np.real(T) 


# psi, diff_psi, xi and diff_xi are auxiliary functions for calculating the Mie coefficients
 
def psi(n, x):
    return x * sps.spherical_jn(n, x, 0)

def diff_psi(n, x):
    return sps.spherical_jn(n, x, 0) + x * sps.spherical_jn(n, x, 1)

def xi(n, x):
    return x * (sps.spherical_jn(n, x, 0) + 1j * sps.spherical_yn(n, x, 0))

def diff_xi(n, x):
    return (sps.spherical_jn(n, x, 0) + 1j * sps.spherical_yn(n, x, 0)) + x * (sps.spherical_jn(n, x, 1) + 1j * sps.spherical_yn(n, x, 1))


# Mie_n: Function that calculates the Mie coefficients.
#
# Inputs:
#
# "k0" is wavevector in vacuum.
#
# "R" is the particle radius.
#
# "m_p" is the particle refractive index.
#
# "m_bg" is the background refractive index.
#
# "order" is the harmonic number order (integer number).
#
# Outpus:
#
# "an" and "bn" are the Mie coefficients or order "n".

def Mie_n(k0, R, m_p, m_bg, order):

    alpha = k0 * R * m_bg
    beta = k0 * R * m_p
    mt = m_p / m_bg

    an = (mt * diff_psi(order, alpha) * psi(order, beta) - psi(order, alpha) * diff_psi(order,beta)) / (mt * diff_xi(order, alpha) * psi(order, beta) - xi(order, alpha) * diff_psi(order, beta))

    bn = (mt * psi(order, alpha) * diff_psi(order, beta) - diff_psi(order, alpha) * psi(order,beta)) / (mt * xi(order, alpha) * diff_psi(order, beta) - diff_xi(order, alpha) * psi(order, beta))

    return an, bn


# "get_alpha_Mie" calculates the polarizability using Mie theory.
#
# Inputs: 
#
# "ko" is wavevector in vacuum.
#
# "R_p" is the particle radius.
#
# "n_p" is the particle refractive index.
#
# "n_b" is the background refractive index.
#
# Outpus:
#
# "alp" is a "6 by 6" matrix with the polarizability of the particle

def get_alpha_Mie(k0, R_p, n_p, n_b):

    k = k0*n_b

    a1, b1 = Mie_n(k0, R_p, n_p, n_b, 1)
    alpha_e = 1j*(6*np.pi)/(k**3)*a1
    alpha_m = 1j*(6*np.pi)/(k**3)*b1 
    id3 = np.eye(3)
    alp = np.zeros((6,6), dtype = 'complex_')  
    alp[0:3,0:3] = id3*alpha_e
    alp[3:6,3:6] = id3*alpha_m

    return alp   
    

# "get_RT" calculates the reflectance and transmittance (RT) for TE and TM incidence for a 
# given metasurface.
#
# Inputs: 
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "R_p" is the particle radius.
#
# "n_p" is the particle refractive index.
#
# "k" is wavevector in the background media (It must be a np.array).
#
# "theta" is the incident angle (in rads) of the incoming plane wave ("theta = 0" is normal incidence). (It must be a np.array).
# "alpha2" is the polarization angle in the "xy" plane (It must be a scalar).
# With this angles, the incident wavevector is defined as:
# 	kx = k*np.sin(theta)*np.cos(alpha2)
#       ky = k*np.sin(theta)*np.sin(alpha2)
#
# "n" and "m" is the calculated diffraction order ("n = m = 0" is the specular mode).
#
# Outputs:
#
# The output are the reflectance and transmittance for different incident
# polarizations (RTM, TTM, RTE, TTE). 

def get_RT(a,b,R_p,n_p,k,theta, alpha2=0, th=np.pi/2, n=0, m=0):

    R_TM = np.zeros((k.shape[0],theta.shape[0]))
    T_TM = np.copy(R_TM)
    R_TE = np.copy(R_TM)
    T_TE = np.copy(R_TM)

    for i in range(k.shape[0]):
        alp = get_alpha_Mie(k[i],R_p,n_p,1)
        for j in range(theta.shape[0]):

            ki = k[i]
            thetaj = theta[j]
            kxi = ki*np.sin(thetaj)*np.cos(alpha2)
            kyi = ki*np.sin(thetaj)*np.sin(alpha2)

            Gb = GbCalc_1UC(a, b, ki, kxi, kyi , th, 100)
            RT = calc_RT(a, b, Gb, alp, ki, kxi, kyi, th, n , m)

            R_TM[i,j] = RT[0]
            T_TM[i,j] = RT[2]
            R_TE[i,j] = RT[1]
            T_TE[i,j] = RT[3]

    return R_TM, R_TE, T_TM, T_TE


# "get_rt" calculates the complex reflection and transmission (RT) for TE and TM incidence for a 
# given metasurface.
#
# Inputs: 
#
# "a" and "b" are the lattice constant along the "x" axis and "sin(th)y +
# cos(th)x" axis. For example, the rectangular lattice is recovered for 
# "th = pi/2", while "th = pi/3" and "a = b" is a triangular lattice.
#
# "R_p" is the particle radius.
#
# "n_p" is the particle refractive index.
#
# "k" is wavevector in the background media (It must by a np.array).
#
# "theta" is the incident angle (in rads) of the incoming plane wave ("theta = 0" is normal incidence). (It must by a np.array).
# "alpha2" is the polarization angle in the "xy" plane (It must by a scalar).
# With this angles, the incident wavevector is defined as:
# 	kx = k*np.sin(theta)*np.cos(alpha2)
#       ky = k*np.sin(theta)*np.sin(alpha2)
#
# Outputs:
#
# The output are the complex reflection and transmissionfor different incident.
# polarizations (rTM, tTM, rTE, tTE). 

def get_rt(a,b,R_p,n_p,k,theta, alpha2=0, th=np.pi/2):

    r_TM = np.zeros((k.shape[0],theta.shape[0]), dtype = 'complex_')
    t_TM = np.copy(r_TM)
    r_TE = np.copy(r_TM)
    t_TE = np.copy(r_TM)

    for i in range(k.shape[0]):
        alp = get_alpha_Mie(k[i],R_p,n_p,1)
        for j in range(theta.shape[0]):

            ki = k[i]
            thetaj = theta[j]
            kxi = ki*np.sin(thetaj)*np.cos(alpha2)
            kyi = ki*np.sin(thetaj)*np.sin(alpha2)

            Gb = GbCalc_1UC(a, b, ki, kxi, kyi , th, 100)
            rt = calc_rt(a, b, Gb, alp, ki, kxi, kyi, th)

            r_TM[i,j] = rt[0]
            t_TM[i,j] = rt[2]
            r_TE[i,j] = rt[1]
            t_TE[i,j] = rt[3]

    return r_TM, r_TE, t_TM, t_TE 
    
