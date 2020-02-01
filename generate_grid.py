import numpy as np
from global_constants import c, max_frequency


def determine_grid_spacing(device_width, layer_permittivities, layer_permeabilities):

    delta_lambda = determine_wavlength_resolution(layer_permittivities, layer_permeabilities)
    delta_d = determine_device_resolution(device_width)
    grid_step_size = snap_grid(device_width, delta_lambda, delta_d)

    return grid_step_size


def determine_wavlength_resolution(layer_permittivities, layer_permeabilities):
    # Wavelength resolution
    wavelength_resolution = 20  # Number of points to resolve a wave with
    max_index_of_refraction = np.max(np.sqrt(layer_permittivities*layer_permeabilities))
    lambda_min = c/(max_frequency*max_index_of_refraction)
    delta_lambda = lambda_min/wavelength_resolution

    return delta_lambda


def determine_device_resolution(device_width):
    # Structure resolution
    device_resolution = 4  # Number of points to resolve device geometry witih
    delta_d = device_width/device_resolution

    return delta_d


def snap_grid(device_width, delta_lambda, delta_d):
    # Grid resolution
    grid_step_size_unsnapped = min(delta_lambda, delta_d)
    grid_size_unsnapped = device_width/grid_step_size_unsnapped
    device_size = round_cells_up(grid_size_unsnapped)
    grid_step_size = device_width/device_size

    return grid_step_size


def round_cells_up(unrounded_number_of_cells):
    return int(np.ceil(unrounded_number_of_cells))


def compute_grid_size(layer_sizes):
    num_reflection_cells = 1
    num_source_cells = 1
    num_transmission_cells = 1

    full_grid_size = num_reflection_cells + num_source_cells + sum(layer_sizes) + num_transmission_cells

    return full_grid_size


def generate_grid_1D(full_grid_size, layer_sizes, epsilons, mus):
    epsilon_r = np.ones(full_grid_size)
    mu_r = np.ones(full_grid_size)

    # Fill grid with device layer by layer
    for i, _ in enumerate(layer_sizes):
        layer_start_index, layer_end_index = compute_layer_start_and_end_indices(layer_sizes, i)
        
        epsilon_r[layer_start_index:layer_end_index] = epsilons[i]
        mu_r[layer_start_index:layer_end_index] = mus[i]

    return (epsilon_r, mu_r)


def compute_layer_start_and_end_indices(layer_sizes, i):
    offset = 2  # num_reflection_cells + num_source_cells
    layer_start_index = offset + sum(layer_sizes[:i])
    layer_end_index = offset + sum(layer_sizes[:i+2])

    return (layer_start_index, layer_end_index)