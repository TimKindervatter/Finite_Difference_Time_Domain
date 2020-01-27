#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define problem
c = 299792458  # Speed of light in m/s
fmax = 1e9  # Hz
d = 0.3048  # Device width in meters
device_permittivity = 6.0
device_permeability = 2.0

# Wavelength resolution
N_lambda = 20  # Number of points to resolve a wave with
nmax = np.sqrt(device_permittivity*device_permeability)
lambda_min = c/(fmax*nmax)
delta_lambda = lambda_min/N_lambda

# Structure resolution
N_d = 4  # Number of points to resolve device geometry witih
delta_d = d/N_d

# Grid resolution
dz_prime = min(delta_lambda, delta_d)
N_prime = d/dz_prime
N = int(np.ceil(N_prime))
dz = d/N

# Determine grid size
spacer_region_size = 10
Nz = N + 2*spacer_region_size + 3

# Determine position of device in grid
num_reflection_cells = 1
num_source_cells = 1
device_start_index = num_reflection_cells + num_source_cells + spacer_region_size
device_end_index = device_start_index + N - 1

# Set device material parameters
epsilon_r = np.ones(Nz)
epsilon_r[device_start_index:device_end_index + 1] = device_permittivity
mu_r = np.ones(Nz)
mu_r[device_start_index:device_end_index + 1] = device_permeability
n = np.sqrt(epsilon_r*mu_r)

# Compute time step
n_bc = 1.0
dt = n_bc*dz/(2*c)

# Compute source parameters
nzsrc = 1
tau = 0.5/fmax
t0 = 6*tau

# Compute number of time steps
t_prop = nmax*Nz*dz/c
total_runtime = 12*tau + 5*t_prop
steps = int(np.ceil(total_runtime/dt))

# Compute source functions for Ey/Hx mode
t = np.arange(0, steps)*dt
A = np.sqrt(epsilon_r[nzsrc]/mu_r[nzsrc])
deltat = n[nzsrc]*dz/(2*c) + dt/2

Eysrc = np.exp(-((t - t0)/tau)**2)
Hxsrc = -A*np.exp(-((t - t0 + deltat)/tau)**2)

# Set up grid
zmin = 0
zmax = Nz*dz
z = np.linspace(zmin, zmax, Nz)

# Initialize update coeffients
mEy = (c*dt)/epsilon_r
mHx = (c*dt)/mu_r

# Initialize field values
Ey = np.zeros(Nz)
Hx = np.zeros(Nz)

h2 = 0
h1 = 0
e2 = 0
e1 = 0

# Initialize Fourier Transforms
Nfreq = 100
freq = np.linspace(0, fmax, Nfreq)
K = np.exp(-1j*2*np.pi*dt*freq)

# Initialize Fourier transforms for reflected and transmitted fields
reflected_fourier = np.zeros(Nfreq, dtype=complex)
transmitted_fourier = np.zeros(Nfreq, dtype=complex)
source_fourier = np.zeros(Nfreq, dtype=complex)

# Initialize plot
fig, ax = plt.subplots(nrows=2, ncols=1)

device_width = z[device_end_index] - z[device_start_index]
rectangle = Rectangle((z[device_start_index], -1.5), device_width, 3, facecolor='grey')

for T in range(steps):
    # Record H at boundary
    h2 = h1
    h1 = Hx[0]
    
    # Update H from E
    for nz in range(Nz-1):
        Hx[nz] = Hx[nz] + mHx[nz]*(Ey[nz + 1] - Ey[nz])/dz
    Hx[Nz-1] = Hx[Nz - 1] + mHx[Nz-1]*(e2 - Ey[Nz - 1])/dz

    # Handle H-field source
    Hx[nzsrc-1] = Hx[nzsrc-1] - (mHx[nzsrc-1]/dz)*Eysrc[T]

    # Record E at boundary
    e2 = e1
    e1 = Ey[Nz - 1]

    # Update E from H
    Ey[0] = Ey[0] + mEy[0]*(Hx[0] - h2)/dz
    for nz in range(1, Nz):
        Ey[nz] = Ey[nz] + mEy[nz]*(Hx[nz] - Hx[nz - 1])/dz

    # Handle E-field source
    Ey[nzsrc] = Ey[nzsrc] - (mEy[nzsrc]/dz)*Hxsrc[T]
    # Ey[nzsrc] = Ey[nzsrc] + g(T)

    # Update Fourier transforms
    for f in range(Nfreq):
        reflected_fourier[f] = reflected_fourier[f] + (K[f]**T)*Ey[0]
        transmitted_fourier[f] = transmitted_fourier[f] + (K[f]**T)*Ey[Nz - 1]
        source_fourier[f] = source_fourier[f] + (K[f]**T)*Eysrc[T]

    reflectance = np.abs(reflected_fourier/source_fourier)**2
    transmittance = np.abs(transmitted_fourier/source_fourier)**2
    conservation_of_energy = reflectance + transmittance

    # Visualize fields
    if (T % 10 == 0):
        ax[0].add_patch(rectangle)
        ax[0].plot(z, Ey)
        ax[0].plot(z, Hx)
        ax[0].set_xlim([zmin, zmax])
        ax[0].set_ylim([-1.5, 1.5])

        ax[1].plot(freq, reflectance)
        ax[1].plot(freq, transmittance)
        ax[1].plot(freq, conservation_of_energy)
        ax[1].set_xlim([0, fmax])
        ax[1].set_ylim([0, 1.5])

        plt.pause(1/60)
        ax[0].cla()
        ax[1].cla()

reflected_fourier = reflected_fourier*dt
transmitted_fourier = transmitted_fourier*dt
source_fourier = source_fourier*dt


# %%
