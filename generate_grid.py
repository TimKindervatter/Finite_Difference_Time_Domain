import numpy as np

def compute_grid_size(layer_widths):
    num_reflection_cells = 1
    num_source_cells = 1
    num_transmission_cells = 1

    full_grid_size = num_reflection_cells + num_source_cells + sum(layer_widths) + num_transmission_cells

    return full_grid_size


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