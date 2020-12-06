# control
from control.matlab import lqr

# math
from numpy import sin, cos
import numpy as np

# diff solver
# from scipy.integrate import solve_ivp 
from scipy.integrate import odeint as oi

# plotting / anim
import matplotlib.pyplot as plt
# from matplotlib import patches as pa
# import matplotlib.animation as an

#------------------------------------------------------------------------------
# global static variables------------------------------------------------------

j = complex(0, 1)

# physics
mc = 6
mp = 0.15
g = 9.82
l = 0.282
mu = 0.01

# linearised system in 0
J = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, -g*mp/mc, -mu/mc, 0], 
                [0, g*(mp+mc)/(l*mc), mu/(l*mc), 0]])
B = np.array([[0],[0],[1/mc],[-1/(l*mc)]])

# LQR
Q = np.diag([50, 2, 0.5, 0.5])
R = np.diag([30])
K, _, eigs = lqr(J, B, Q, R)

# time
numSamp = 1000 
t1 = 0
t2 = 10
t = np.linspace(t1, t2, numSamp)

# Initial condition and stabilization point
z0 = [0, 0.1, 0, 0]
zbar = np.array([0, 0, 0, 0])


#------------------------------------------------------------------------------
# System equations-------------------------------------------------------------

# Non stabilised system
def nonLin(t, z):
    z1, z2, z3, z4 = z
    
    z1_ = z3
    z2_ = z4
    z3_ = (-mp*g*cos(z2)*sin(z2)+mp*l*z4**2*sin(z2)-mu*z3) /  \
        (mp+mc-mp*cos(z2)**2)
    z4_ = (cos(z2)*(mp*g*cos(z2)*sin(z2)-mp*l*z4**2*sin(z2)+mu*z3)) / \
        (l*(mp+mc-mp*cos(z2)**2)) + (g*sin(z2)) / (l)
        
    return [z1_, z2_, z3_, z4_]

def lin(t, z):
    return J@z

# stabilised system
def nonLinS(t, z, zBar):
    z1, z2, z3, z4 = z
    
    F = -K@(z-zBar)
    
    z1_ = z3
    z2_ = z4
    z3_ = (-mp*g*cos(z2)*sin(z2)+mp*l*z4**2*sin(z2)+F-mu*z3) /  \
        (mp+mc-mp*cos(z2)**2)
    z4_ = (cos(z2)*(mp*g*cos(z2)*sin(z2)-mp*l*z4**2*sin(z2)-F+mu*z3)) / \
        (l*(mp+mc-mp*cos(z2)**2)) + (g*sin(z2)) / (l)
    
    return [z1_, z2_, z3_, z4_]

def linS(t, z, zBar):
    A = ((J-B@K)@(z-zBar))
    a1 = A[0, 0]
    a2 = A[0, 1]
    a3 = A[0, 2]
    a4 = A[0, 3]
    return np.array([a1, a2, a3, a4])

#------------------------------------------------------------------------------
# Plooting tools

# plot lims
def lims(array, perc = 0.05):
    delta = max(array)-min(array)
    min_ = min(array)-perc*delta
    max_ = max(array)+perc*delta
    
    return (min_, max_)


# Plotter
def plotter(z0_ = z0, zstab_ = zbar, stab = False):
    """
    Simulerer og plotter pendulet

    Parameters
    ----------
    z0_ :
        Start betingelsen der simuleres fra. The default is z0.
    zstab_ :
        Det punkt systemet skal stabiliseres i. The default is zbar.
    stab :
        Hvorvidt systemet skal stabiliseres. The default is False.

    """
    
    
    # simulating
    if stab:
        zNonLin = oi(nonLinS, z0_, t, tfirst = True, args = (zstab_,))
        zLin = oi(linS, z0_, t, tfirst = True, args = (zstab_,))
    else:
        zNonLin = oi(nonLin, z0_, t, tfirst = True)
        zLin = oi(lin, z0_, t, tfirst = True)
    
    # plotting
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])
    ax = ax.ravel()
    fig.subplots_adjust(wspace = 0.4, hspace=0.4)
    
    # fancy stuff
    nams_ = [r'$x_c\quad [m]$', r'$\theta\quad [rad]$',
             r'$\.x_c\quad [\frac{m}{s}]$', r'$\.\theta\quad [\frac{rad}{s}]$']
    
    if stab:
        fig.suptitle("Stabilise in " + str(zstab_)
                     + r" with $z_0 = $" + str(z0_))
    else:
        fig.suptitle(r"No control - with $z_0 = $" + str(z0_))
    
    for i in range(4):
        ax[i].set_ylim(lims(zNonLin[:,i], perc = 0.1))
        ax[i].grid()
        ax[i].set_ylabel(nams_[i])
        ax[i].set_xlabel(r'$t\quad [s]$')
        
        ax[i].plot(t, zNonLin[:,i], '-r')
        ax[i].plot(t, zLin[:,i], '-b', ls = '-')
    
    plt.show()

#------------------------------------------------------------------------------
plotter()

plotter(z0_ = [1, 1, -1, -1], zstab_ = [0, 0, 0, 0], stab = True)