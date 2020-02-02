import numpy as np
import generate_grid as grid
import matplotlib.pyplot as plt
import utils

c = 299792458  # Speed of light in m/s


class ProblemSetup:
    device = None
    max_frequency = None
    figure = None
    axes = None
    rectangles = None

    def __init__(self, device_name):
        if device_name == "Slab":
            self.max_frequency = 1e9  # Hz
            self.device = Slab(self.max_frequency)
        elif device_name == "AntiReflectionLayer":
            self.max_frequency = 5e9  # Hz
            self.device = AntiReflectionLayer(self.max_frequency)
            
        # Initialize plot
        self.figure, self.axes = plt.subplots(nrows=2, ncols=1)
        self.rectangles = utils.create_layer_shadings(self.device)


class Device:
    layer_widths = None
    layer_permittivities = None
    layer_permeabilities = None
    layer_refractive_indices = None
    grid_resolution = None
    layer_sizes = None
    full_grid_size = None
    epsilon_r = None
    mu_r = None
    index_of_refraction = None
    grid = None

    def __init__(self, max_frequency):
        self.grid_resolution = grid.determine_grid_spacing(self, max_frequency)
        self.layer_sizes = grid.compute_layer_sizes(self)
        self.full_grid_size = grid.compute_grid_size(self.layer_sizes)
        self.epsilon_r, self.mu_r = grid.generate_material_grid(self)
        self.index_of_refraction = np.sqrt(self.epsilon_r*self.mu_r)
        self.grid = grid.generate_grid_1D(self)


class Slab(Device):
    def __init__(self, max_frequency):
        
        device_width = 0.3048
        spacer_region_width = 0.3048/7.1

        self.layer_widths = [spacer_region_width, device_width, spacer_region_width]  # Layer widths in meters
        self.layer_permittivities = np.array([1.0, 6.0, 1.0])
        self.layer_permeabilities = np.array([1.0, 2.0, 1.0])
        self.layer_refractive_indices = np.sqrt(self.layer_permittivities*self.layer_permeabilities)
        
        super().__init__(max_frequency)


class AntiReflectionLayer(Device):
    def __init__(self, max_frequency):
        device_width = 0.3048
        spacer_region_width = 0.3048/7.1
        anti_reflection_width = 1.6779e-2

        self.layer_widths = [spacer_region_width, anti_reflection_width, device_width, anti_reflection_width, spacer_region_width]  # Layer widths in meters
        self.layer_permittivities = np.array([1.0, 3.46, 12.0, 3.46, 1.0])
        self.layer_permeabilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.layer_refractive_indices = np.sqrt(self.layer_permittivities*self.layer_permeabilities)
        
        super().__init__(max_frequency)