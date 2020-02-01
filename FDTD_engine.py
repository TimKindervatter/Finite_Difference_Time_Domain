#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Define problem
c = 299792458  # Speed of light in m/s
max_frequency = 1e9  # Hz
device_width = 0.3048  # Device width in meters
layer_permittivities = np.array([1.0, 6.0, 1.0])
layer_permeabilities = np.array([1.0, 2.0, 1.0])

# Wavelength resolution
wavelength_resolution = 20  # Number of points to resolve a wave with
max_index_of_refraction = np.max(np.sqrt(layer_permittivities*layer_permeabilities))
lambda_min = c/(max_frequency*max_index_of_refraction)
delta_lambda = lambda_min/wavelength_resolution

# Structure resolution
device_resolution = 4  # Number of points to resolve device geometry witih
delta_d = device_width/device_resolution

# Grid resolution
grid_step_size_unsnapped = min(delta_lambda, delta_d)
grid_size_unsnapped = device_width/grid_step_size_unsnapped
device_size = int(np.ceil(grid_size_unsnapped))
dz = device_width/device_size

# Determine grid size
spacer_region_size = 10
layer_widths = [spacer_region_size, device_size, spacer_region_size]


def compute_grid_size(layer_widths):
    num_reflection_cells = 1
    num_source_cells = 1
    num_transmission_cells = 1

    full_grid_size = num_reflection_cells + num_source_cells + sum(layer_widths) + num_transmission_cells

    return full_grid_size


Nz = compute_grid_size(layer_widths)


def generate_grid_1D(full_grid_size, layer_widths, epsilons, mus):
    epsilon_r = np.ones(full_grid_size)
    mu_r = np.ones(full_grid_size)

    # Fill grid with device layer by layer
    for i, _ in enumerate(layer_widths):
        offset = 2  # num_reflection_cells + num_source_cells
        layer_start_index = offset + sum(layer_widths[:i])
        layer_end_index = offset + sum(layer_widths[:i+2])

        epsilon_r[layer_start_index:layer_end_index] = epsilons[i]
        mu_r[layer_start_index:layer_end_index] = mus[i]

    return (epsilon_r, mu_r)

    # device_start_index = num_reflection_cells + num_source_cells + spacer_region_size
    # device_end_index = device_start_index + device_size - 1

    # # Set device material parameters
    # epsilon_r = np.ones(Nz)
    # epsilon_r[device_start_index:device_end_index + 1] = device_permittivity
    # mu_r = np.ones(Nz)
    # mu_r[device_start_index:device_end_index + 1] = device_permeability
    

epsilon_r, mu_r = generate_grid_1D(Nz, layer_widths, layer_permittivities, layer_permeabilities)
n = np.sqrt(epsilon_r*mu_r)

# Compute time step
boundary_refractive_index = 1.0
dt = boundary_refractive_index*dz/(2*c)

# Compute source parameters
nzsrc = 1
tau = 0.5/max_frequency
t0 = 6*tau

# Compute number of time steps
t_prop = max_index_of_refraction*Nz*dz/c
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
freq = np.linspace(0, max_frequency, Nfreq)
K = np.exp(-1j*2*np.pi*dt*freq)

# Initialize Fourier transforms for reflected and transmitted fields
reflected_fourier = np.zeros(Nfreq, dtype=complex)
transmitted_fourier = np.zeros(Nfreq, dtype=complex)
source_fourier = np.zeros(Nfreq, dtype=complex)

# Initialize plot
fig, ax = plt.subplots(nrows=2, ncols=1)

device_start_index = layer_widths[0] + 2
device_end_index = layer_widths[0] + layer_widths[1] + 2
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
        ax[1].set_xlim([0, max_frequency])
        ax[1].set_ylim([0, 1.5])

        plt.pause(1/60)
        ax[0].cla()
        ax[1].cla()

reflected_fourier = reflected_fourier*dt
transmitted_fourier = transmitted_fourier*dt
source_fourier = source_fourier*dt


# %%
