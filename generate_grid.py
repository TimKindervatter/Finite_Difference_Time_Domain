import numpy as np
from config import c
import config


def determine_grid_spacing(device, max_frequency):
    delta_lambda = determine_wavlength_resolution(device.layer_permittivities, device.layer_permeabilities, max_frequency)
    critical_dimension = np.min(device.layer_widths)
    delta_d = determine_device_resolution(critical_dimension)
    grid_step_size = snap_grid(critical_dimension, delta_lambda, delta_d)

    return grid_step_size


def determine_wavlength_resolution(layer_permittivities, layer_permeabilities, max_frequency):
    # Wavelength resolution
    wavelength_resolution = 20  # Number of points to resolve a wave with
    max_index_of_refraction = np.max(np.sqrt(layer_permittivities*layer_permeabilities))
    lambda_min = c/(max_frequency*max_index_of_refraction)
    delta_lambda = lambda_min/wavelength_resolution

    return delta_lambda


def determine_device_resolution(critical_dimension):
    # Structure resolution
    device_resolution = 4  # Number of points to resolve device geometry witih
    delta_d = critical_dimension/device_resolution

    return delta_d


def snap_grid(critical_dimension, delta_lambda, delta_d):
    # Grid resolution
    grid_step_size_unsnapped = min(delta_lambda, delta_d)
    grid_size_unsnapped = critical_dimension/grid_step_size_unsnapped
    device_size = round_cells_up(grid_size_unsnapped)
    grid_step_size = critical_dimension/device_size

    return grid_step_size


def round_cells_up(unrounded_number_of_cells):
    return int(np.ceil(unrounded_number_of_cells))


def compute_layer_sizes(device):
    layer_widths = device.layer_widths
    dz = device.grid_resolution

    # Determine grid size
    layer_sizes = []
    for layer_width in layer_widths:
        layer_size = int(np.ceil(layer_width/dz))
        layer_sizes.append(layer_size)

    # full_grid_size = compute_grid_size(layer_sizes)

    return layer_sizes


def compute_grid_size(layer_sizes):
    num_reflection_cells = 1
    num_source_cells = 1
    num_transmission_cells = 1

    full_grid_size = num_reflection_cells + num_source_cells + sum(layer_sizes) + num_transmission_cells

    return full_grid_size


def generate_material_grid(device):
    epsilon_r = np.ones(device.full_grid_size)
    mu_r = np.ones(device.full_grid_size)

    # Fill grid with device layer by layer
    for i, _ in enumerate(device.layer_sizes):
        layer_start_index, layer_end_index = compute_layer_start_and_end_indices(device.layer_sizes, i)

        epsilon_r[layer_start_index:layer_end_index] = device.layer_permittivities[i]
        mu_r[layer_start_index:layer_end_index] = device.layer_permeabilities[i]

    return (epsilon_r, mu_r)


def generate_grid_1D(device):
    Nz = device.full_grid_size
    dz = device.grid_resolution

    # Set up grid
    zmin = 0
    zmax = Nz*dz
    grid = np.linspace(zmin, zmax, Nz)

    return grid


def compute_layer_start_and_end_indices(layer_sizes, i):
    offset = 2  # num_reflection_cells + num_source_cells
    layer_start_index = offset + sum(layer_sizes[:i])
    layer_end_index = offset + sum(layer_sizes[:i+2])

    return (layer_start_index, layer_end_index)