import numpy as np
import device as dv
from matplotlib.patches import Rectangle


def create_layer_shadings(device):
    full_grid = device.grid
    layer_sizes = device.layer_sizes
    layer_refractive_indices = device.layer_refractive_indices

    rectangles = []
    normalized_refractive_indices = (layer_refractive_indices - np.min(layer_refractive_indices))/(np.max(layer_refractive_indices) - np.min(layer_refractive_indices))

    for i, layer_size in enumerate(layer_sizes):
        layer_start_index, _ = device.compute_layer_start_and_end_indices(layer_sizes, i)

        layer_color = str(1 - 0.6*normalized_refractive_indices[i])
        rectangle = Rectangle((full_grid[layer_start_index], -1.5), layer_size, 3, facecolor=layer_color)
        rectangles.append(rectangle)

    return rectangles