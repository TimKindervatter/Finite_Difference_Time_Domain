#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
import config
from config import c
from timeit import default_timer as timer
from numba import njit


def FDTD_engine(plot=False):
    # Define problem
    device_name = "AntiReflectionLayer"
    problem_instance = config.ProblemInstance(device_name)

    max_frequency = problem_instance.max_frequency

    Nz = problem_instance.device.full_grid_size
    dz = problem_instance.device.grid_resolution
    z = problem_instance.device.grid

    epsilon_r = problem_instance.device.epsilon_r
    mu_r = problem_instance.device.mu_r
    n = problem_instance.device.index_of_refraction

    dt = problem_instance.time_step

    source_location = 1

    CW = False
    pulse = True
    debug = False

    # Compute source parameters
    if CW:
        tau = 3/max_frequency
        t0 = 3*tau
    if pulse:
        tau = 0.5/max_frequency
        t0 = 6*tau

    # Compute number of time steps
    max_index_of_refraction = np.max(n)
    t_prop = max_index_of_refraction*Nz*dz/c

    # TODO: Make the coefficient out front a problem_instance member
    total_runtime = 1*(12*tau + 5*t_prop)
    steps = int(np.ceil(total_runtime/dt))

    # Compute source functions for Ey/Hx mode
    t = np.arange(0, steps)*dt
    A = np.sqrt(epsilon_r[source_location]/mu_r[source_location])
    deltat = n[source_location]*dz/(2*c) + dt/2

    if CW:
        f_op = 1e9

        g_E = np.sin(2*np.pi*f_op*t) + np.finfo(float).eps
        g_H = -np.sin(2*np.pi*f_op*t) + np.finfo(float).eps

        Eysrc = g_E*np.ones(len(t))
        Hxsrc = g_H*np.ones(len(t))

        E_envelope = np.exp(-((t - t0)/tau)**2)
        H_envelope = -A*np.exp(-((t - t0 + deltat)/tau)**2)

        Eysrc[t < t0] *= np.exp(-((t[t < t0] - t0)/tau)**2)
        Hxsrc[t < t0] *= A*np.exp(-((t[t < t0] - t0 + deltat)/tau)**2)

    elif pulse:
        E_envelope = np.exp(-((t - t0)/tau)**2)
        H_envelope = -A*np.exp(-((t - t0 + deltat)/tau)**2)

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

    fourier_transform_manager = problem_instance.fourier_transform_manager

    for T in range(steps):
        # Record H at boundary
        h2 = h1
        h1 = Hx[0]
        
        # Update H from E
        update_H(Hx, Ey, mHx, dz, Nz)
        Hx[Nz-1] = Hx[Nz - 1] + mHx[Nz-1]*(e2 - Ey[Nz - 1])/dz

        # Handle H-field source
        Hx[source_location-1] = Hx[source_location-1] - (mHx[source_location-1]/dz)*Eysrc[T]

        # Record E at boundary
        e2 = e1
        e1 = Ey[Nz - 1]

        # Update E from H
        Ey[0] = Ey[0] + mEy[0]*(Hx[0] - h2)/dz
        update_E(Ey, Hx, mEy, dz, Nz)

        # Handle E-field source
        Ey[source_location] = Ey[source_location] - (mEy[source_location]/dz)*Hxsrc[T]
        # Ey[nzsrc] = Ey[nzsrc] + g(T)

        fourier_transform_manager.update_fourier_transforms(T, Ey, Eysrc, Nz)

        if plot:
            utils.update_plot(T, z, Ey, Hx, problem_instance)
            if debug:
                utils.update_source_plot(problem_instance, T, t, Eysrc, Hxsrc)
            plt.pause(0.0001)

    fourier_transform_manager.finalize_fourier_transforms()

    return (fourier_transform_manager.reflected_fourier, fourier_transform_manager.transmitted_fourier, fourier_transform_manager.source_fourier, fourier_transform_manager.conservation_of_energy)


@njit(parallel=True)
def update_H(Hx, Ey, mHx, dz, Nz):
    for nz in range(Nz-1):
        Hx[nz] = Hx[nz] + mHx[nz]*(Ey[nz + 1] - Ey[nz])/dz


def update_E(Ey, Hx, mEy, dz, Nz):
    for nz in range(1, Nz):
        Ey[nz] = Ey[nz] + mEy[nz]*(Hx[nz] - Hx[nz - 1])/dz


if __name__ == "__main__":
    FDTD_engine(True)