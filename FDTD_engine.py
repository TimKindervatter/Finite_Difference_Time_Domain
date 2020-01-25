#%%
import numpy as np
import matplotlib.pyplot as plt

c = 299792458   # Speed of light in m/s
steps = 200     # Number of iterations

Nres = 1       # Number of points to resolve a wave with
dz = 1/Nres
Nz = 100        # Number of grid points along the z-axis


dmin = 1        # Size of smallest feature in geomtetry
NDres = 4       # Number of points to resolve a dimension with

nmax = 1.0      # Maximum refractive index

# _lambda = np.linspace(10, 100, 1000)  # Wavelengths to simulate

# Compute default grid resolution
# dz1 = np.min(_lambda)/nmax/Nres
# dz2 = dmin/NDres
# dz = min(dz1, dz2)


# Snap grid to critical dimensions
# dc = dz1
# N = np.ceil(dc/dz)
# dz = dc/N

zmin = 0
zmax = Nz*dz
z = np.linspace(zmin, zmax, Nz)

epsilon_r = np.ones(Nz)
mu_r = np.ones(Nz)
nbc = 1  # Refractive index of boundary

dt = nbc*dz/(2*c)


#  Pulse parameters
# _lambda_min = min(_lambda)
fmax = 0.1
nzsrc = Nz//2
tau = 0.5/fmax
t0 = 5*tau
g = lambda t: np.exp(-((t - t0)/tau)**2)
A = np.sqrt(epsilon_r[nzsrc]/mu_r[nzsrc])
deltat = dt + dt/2

Eysrc = lambda t: g(t)
Hxsrc = lambda t: -A*g(T + deltat)

mEy = (c*dt)/epsilon_r
mHx = (c*dt)/mu_r

Ey = np.zeros(Nz)
Hx = np.zeros(Nz)

h3 = 0
h2 = 0
h1 = 0
e3 = 0
e2 = 0
e1 = 0

plt.figure(1)

for T in range(steps):
    # h3 = h2
    h2 = h1
    h1 = Hx[0]
    for nz in range(Nz-1):
        Hx[nz] = Hx[nz] + mHx[nz]*(Ey[nz + 1] - Ey[nz])/dz
    Hx[Nz-1] = Hx[Nz - 1] + mHx[Nz-1]*(e2 - Ey[Nz - 1])/dz

    Hx[nzsrc-1] = Hx[nzsrc-1] - (mHx[nzsrc-1]/dz)*Eysrc(T)

    # e3 = e2
    e2 = e1
    e1 = Ey[Nz - 1]
    Ey[0] = Ey[0] + mEy[0]*(Hx[0] - h2)/dz
    for nz in range(1, Nz):
        Ey[nz] = Ey[nz] + mEy[nz]*(Hx[nz] - Hx[nz - 1])/dz


    Ey[nzsrc] = Ey[nzsrc] - (mEy[nzsrc]/dz)*Hxsrc(T)
    # Ey[nzsrc] = Ey[nzsrc] + g(T)

    plt.plot(z, Ey)
    plt.plot(z, Hx)
    axes = plt.gca()
    axes.set_xlim([zmin, zmax])
    axes.set_ylim([-1.1, 1.1])
    plt.pause(1/60)
    plt.cla()


# %%
