from control.matlab import lqr

from numpy import sin, cos
import numpy as np

from scipy.integrate import solve_ivp
# from scipy.integrate import odeint as oi

import matplotlib.pyplot as plt
# from matplotlib import patches as pa
# import matplotlib.animation as an


# System variabler-------------------------------------------------------------
j = complex(0, 1)

mc = 6
mp = 0.15
g = 9.82
l = 0.282
mu = 0.01
F = 0 # change this in the functions describing the motion


# linearised system
J1 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, -g*mp/mc, -mu/mc, 0], 
                [0, g*(mp+mc)/(l*mc), mu/(l*mc), 0]])
# J2 = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, -g*mp/mc, -mu/mc, 0],
#                [0, -g*(mp+mc)/(l*mc), -mu/(l*mc), 0]])
B = np.array([[0],[0],[1/mc],[-1/(l*mc)]])

# weight matrices for LQR
Q = np.diag([50, 2, 0.5, 0.5])
R = np.diag([30])

F1, a_, eigs = lqr(J1, B, Q, R)
F1 = np.array(F1)

# Simulering Variabler---------------------------------------------------------
z0 = [0, 0.1, 0, 0]
t0 = 0
t1 = 15
N = 100
t = np.linspace(t0, t1, N)

stablePoint = [0, 0, 0, 0]
forces = []

# diff eq----------------------------------------------------------------------
def nonLin(z, t):
    z1, z2, z3, z4 = z
    
    z1_ = z3
    z2_ = z4
    z3_ = (-mp*g*cos(z2)*sin(z2)+mp*l*z4**2*sin(z2)+F-mu*z3) /  \
        (mp+mc-mp*cos(z2)**2)
    z4_ = (cos(z2)*(mp*g*cos(z2)*sin(z2)-mp*l*z4**2*sin(z2)-F+mu*z3)) / \
        (l*(mp+mc-mp*cos(z2)**2)) + (g*sin(z2)) / (l)
    
    return [z1_, z2_, z3_, z4_]

def lin(t, z):# lin omkring theta = 0
    forces.append((F1@(z-stablePoint))[0])
    
    return ((J1-B@F1)@(z-stablePoint))


# Non stabilised pendulum with F-----------------------------------------------
'''
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

sol = oi(nonLin, z0, t)

pos = [sol[:,0], sol[:,1]] #her skal akserne ændres
dot = [sol[:,2], sol[:,3]]

for i in range(2):
    ax[0][i].plot(pos[i])
    ax[0][i].set(xlabel = "z" + str(i+1))

for i in range(2):
    ax[1][i].plot(dot[i])
    ax[1][i].set(xlabel = "z" + str(i+3))

plt.show()
'''
# Stabilised pendulum----------------------------------------------------------

figS, axS = plt.subplots(2, 2)
figS.tight_layout()

solS = solve_ivp(lin, (t0, t1), z0)

t_ = np.linspace(t0, t1, len(solS.y[0,:]))

posS = [solS.y[0,:], solS.y[1,:]]
dotS = [solS.y[2,:], solS.y[3,:]]

for i in range(2):
    axS[0][i].plot(t_, posS[i])
    axS[0][i].set(xlabel = "z" + str(i+1))

for i in range(2):
    axS[1][i].plot(t_, dotS[i])
    axS[1][i].set(xlabel = "z" + str(i+3))

plt.show()

print("F: ", min(forces), max(forces))
print("z1: ", min(posS[0]), max(posS[0]))

#Animation---------------------------------------------------------------------
'''
fig = plt.figure(figsize=(5, 5), facecolor='w')
ax = fig.add_subplot(1, 1, 1)
plt.rcParams['font.size'] = 15
ax.set_xlim([-0.5, 0.5])
lns = []
for i in range(len(posS[0])):
    ln, = ax.plot([wsol, np.sin(wsol[i, 1])*l+wsol[i,0]], [0, np.cos(wsol[i, 1])*l],
                  color='k', lw=2)
    
    tm = ax.text(-1, 0.9, 'time = %.1fs' % t1[i])
    lns.append([ln, tm])
ax.set_aspect('equal', 'datalim')

ax.grid()
ani = animation.ArtistAnimation(fig, lns, interval=50)

fn = 'test_pendul'
ani.save(fn+'.gif',writer='imagemagick',fps=20)
'''