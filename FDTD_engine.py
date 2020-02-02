#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
import config
from config import c


def FDTD_engine(plot=False):
    # Define problem
    device_name = "Slab"
    problem_instance = config.ProblemSetup(device_name)

    max_frequency = problem_instance.max_frequency

    Nz = problem_instance.device.full_grid_size
    dz = problem_instance.device.grid_resolution
    z = problem_instance.device.grid

    epsilon_r = problem_instance.device.epsilon_r
    mu_r = problem_instance.device.mu_r
    n = problem_instance.device.index_of_refraction

    fig = problem_instance.figure
    ax = problem_instance.axes
    rectangles = problem_instance.rectangles

    # Compute time step
    boundary_refractive_index = 1.0
    dt = boundary_refractive_index*dz/(2*c)

    # Compute source parameters
    nzsrc = 1
    tau = 0.5/max_frequency
    t0 = 6*tau

    # Compute number of time steps
    max_index_of_refraction = np.max(n)
    t_prop = max_index_of_refraction*Nz*dz/c

    total_runtime = 12*tau + 5*t_prop
    steps = int(np.ceil(total_runtime/dt))

    # Compute source functions for Ey/Hx mode
    t = np.arange(0, steps)*dt
    A = np.sqrt(epsilon_r[nzsrc]/mu_r[nzsrc])
    deltat = n[nzsrc]*dz/(2*c) + dt/2

    Eysrc = np.exp(-((t - t0)/tau)**2)
    Hxsrc = -A*np.exp(-((t - t0 + deltat)/tau)**2)

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
    Nfreq = 500
    freq = np.linspace(0, max_frequency, Nfreq)
    K = np.exp(-1j*2*np.pi*dt*freq)

    # Initialize Fourier transforms for reflected and transmitted fields
    reflected_fourier = np.zeros(Nfreq, dtype=complex)
    transmitted_fourier = np.zeros(Nfreq, dtype=complex)
    source_fourier = np.zeros(Nfreq, dtype=complex)

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
        #TODO: Parallelize Fourier computation
        for f in range(Nfreq):
            reflected_fourier[f] = reflected_fourier[f] + (K[f]**T)*Ey[0]
            transmitted_fourier[f] = transmitted_fourier[f] + (K[f]**T)*Ey[Nz - 1]
            source_fourier[f] = source_fourier[f] + (K[f]**T)*Eysrc[T]

        reflectance = np.abs(reflected_fourier/source_fourier)**2
        transmittance = np.abs(transmitted_fourier/source_fourier)**2
        conservation_of_energy = reflectance + transmittance

        if plot:
            # Visualize fields
            if (T % 20 == 0):
                for rectangle in rectangles:
                    ax[0].add_patch(rectangle)
                ax[0].plot(z, Ey)
                ax[0].plot(z, Hx)
                ax[0].set_xlim([z[0], z[-1]])
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

    return (reflected_fourier, transmitted_fourier, source_fourier, conservation_of_energy)


if __name__ == "__main__":
    FDTD_engine(True)