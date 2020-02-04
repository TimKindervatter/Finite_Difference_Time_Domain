import numpy as np
import device as dv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import c


def create_layer_shadings(device):
    full_grid = device.grid
    layer_sizes = device.layer_sizes
    layer_refractive_indices = device.layer_refractive_indices

    rectangles = []
    if (np.max(layer_refractive_indices) - np.min(layer_refractive_indices) != 0):
        normalized_refractive_indices = (layer_refractive_indices - np.min(layer_refractive_indices))/(np.max(layer_refractive_indices) - np.min(layer_refractive_indices))

        for i, layer_size in enumerate(layer_sizes):
            layer_start_index, _ = device.compute_layer_start_and_end_indices(layer_sizes, i)

            layer_color = str(1 - 0.6*normalized_refractive_indices[i])
            rectangle = Rectangle((full_grid[layer_start_index], -1.5), layer_size, 3, facecolor=layer_color)
            rectangles.append(rectangle)

    return rectangles


def update_plot(T, z, Ey, Hx, problem_instance):
    fourier_transform_manager = problem_instance.fourier_transform_manager
    ax = problem_instance.main_axes

    # Visualize fields
    if (T % problem_instance.plot_update_interval == 0):
        ax[0].cla()
        ax[1].cla()

        for rectangle in problem_instance.rectangles:
            ax[0].add_patch(rectangle)
        ax[0].plot(z, Ey)
        ax[0].plot(z, Hx)
        ax[0].set_xlim([z[0], z[-1]])
        ax[0].set_ylim([-1.5, 1.5])

        # TODO: Make plotting style a problem_instance member
        linear = True
        log = False

        if linear:
            ax[1].plot(fourier_transform_manager.freq, fourier_transform_manager.reflectance, label='Reflectance')
            ax[1].plot(fourier_transform_manager.freq, fourier_transform_manager.transmittance, label='Transmittance')
            ax[1].plot(fourier_transform_manager.freq, fourier_transform_manager.conservation_of_energy, label='Conservation')
            ax[1].set_xlim([0, problem_instance.max_frequency])
            ax[1].set_ylim([0, 1.5])
            ax[1].legend()
        if log:
            ax[1].plot(fourier_transform_manager.freq, 10*np.log10(fourier_transform_manager.reflectance), label='Reflectance')
            ax[1].plot(fourier_transform_manager.freq, 10*np.log10(fourier_transform_manager.transmittance), label='Transmittance')
            ax[1].plot(fourier_transform_manager.freq, 10*(fourier_transform_manager.conservation_of_energy), label='Conservation')
            ax[1].set_xlim([2e14, 4e14])
            ax[1].set_ylim([-40, 0])
            ax[1].legend()
        
    
def update_source_plot(problem_instance, T, t, Eysrc, Hxsrc):
    ax = problem_instance.source_axes
    if (T % problem_instance.plot_update_interval == 0):
        ax[0].cla()
        ax[1].cla()

        ax[0].plot(t, Eysrc)
        ax[0].plot(t[T], Eysrc[T], 'ro')
        ax[1].plot(t, Hxsrc)
        ax[1].plot(t[T], Hxsrc[T], 'ro')