import numpy as np
from config import c
import config


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
    boundary_refractive_index = None

    def __init__(self, max_frequency):
        if type(self) is Device:
            raise Exception('Device is an abstract class and cannot be instantiated directly')

        self.layer_refractive_indices = np.sqrt(self.layer_permittivities*self.layer_permeabilities)
        self.grid_resolution = self.determine_grid_spacing(max_frequency)
        self.layer_sizes = self.compute_layer_sizes()
        self.full_grid_size = self.compute_grid_size(self.layer_sizes)
        self.epsilon_r, self.mu_r = self.generate_material_grid()
        self.index_of_refraction = np.sqrt(self.epsilon_r*self.mu_r)
        self.grid = self.generate_grid_1D()

    def determine_grid_spacing(self, max_frequency):
        delta_lambda = self.determine_wavlength_resolution(self.layer_permittivities, self.layer_permeabilities, max_frequency)
        critical_dimension = np.min(self.layer_widths)
        delta_d = self.determine_device_resolution(critical_dimension)
        grid_step_size = self.snap_grid(critical_dimension, delta_lambda, delta_d)

        return grid_step_size

    def determine_wavlength_resolution(self, layer_permittivities, layer_permeabilities, max_frequency):
        # Wavelength resolution
        wavelength_resolution = 20  # Number of points to resolve a wave with
        max_index_of_refraction = np.max(np.sqrt(layer_permittivities*layer_permeabilities))
        lambda_min = c/(max_frequency*max_index_of_refraction)
        delta_lambda = lambda_min/wavelength_resolution

        return delta_lambda

    def determine_device_resolution(self, critical_dimension):
        # Structure resolution
        device_resolution = 4  # Number of points to resolve device geometry witih
        delta_d = critical_dimension/device_resolution

        return delta_d

    def snap_grid(self, critical_dimension, delta_lambda, delta_d):
        # Grid resolution
        grid_step_size_unsnapped = min(delta_lambda, delta_d)
        grid_size_unsnapped = critical_dimension/grid_step_size_unsnapped
        device_size = self.round_cells_up(grid_size_unsnapped)
        grid_step_size = critical_dimension/device_size

        return grid_step_size

    def round_cells_up(self, unrounded_number_of_cells):
        return int(np.ceil(unrounded_number_of_cells))

    def compute_layer_sizes(self):
        layer_widths = self.layer_widths
        dz = self.grid_resolution

        # Determine grid size
        layer_sizes = []
        for layer_width in layer_widths:
            layer_size = int(np.ceil(layer_width/dz))
            layer_sizes.append(layer_size)

        return layer_sizes

    def compute_grid_size(self, layer_sizes):
        num_reflection_cells = 1
        num_source_cells = 1
        num_transmission_cells = 1

        full_grid_size = num_reflection_cells + num_source_cells + sum(layer_sizes) + num_transmission_cells

        return full_grid_size

    def generate_material_grid(self):
        epsilon_r = np.ones(self.full_grid_size)
        mu_r = np.ones(self.full_grid_size)

        # Fill grid with device layer by layer
        for i, _ in enumerate(self.layer_sizes):
            layer_start_index, layer_end_index = self.compute_layer_start_and_end_indices(self.layer_sizes, i)

            epsilon_r[layer_start_index:layer_end_index] = self.layer_permittivities[i]
            mu_r[layer_start_index:layer_end_index] = self.layer_permeabilities[i]

        return (epsilon_r, mu_r)

    def generate_grid_1D(self):
        Nz = self.full_grid_size
        dz = self.grid_resolution

        # Set up grid
        zmin = 0
        zmax = Nz*dz
        grid = np.linspace(zmin, zmax, Nz)

        return grid

    def compute_layer_start_and_end_indices(self, layer_sizes, i):
        offset = 2  # num_reflection_cells + num_source_cells
        layer_start_index = offset + sum(layer_sizes[:i])
        layer_end_index = offset + sum(layer_sizes[:i+2])

        return (layer_start_index, layer_end_index)


class FreeSpace(Device):
    def __init__(self, max_frequency):
        device_width = 2

        self.layer_widths = [device_width]  # Layer widths in meters
        self.layer_permittivities = np.array([1.0])
        self.layer_permeabilities = np.array([1.0])
        
        self.boundary_refractive_index = 1.0
        
        super().__init__(max_frequency)


class Slab(Device):
    def __init__(self, max_frequency):
        
        device_width = 0.3048
        spacer_region_width = 0.3048/7.1

        self.layer_widths = [spacer_region_width, device_width, spacer_region_width]  # Layer widths in meters
        self.layer_permittivities = np.array([1.0, 6.0, 1.0])
        self.layer_permeabilities = np.array([1.0, 2.0, 1.0])
        
        self.boundary_refractive_index = 1.0
        
        super().__init__(max_frequency)


class AntiReflectionLayer(Device):
    def __init__(self, max_frequency):
        device_width = 0.3048
        spacer_region_width = 0.3048/7.1  # Results in 10 grid cells
        anti_reflection_width = 1.6779e-2

        self.layer_widths = [spacer_region_width, anti_reflection_width, device_width, anti_reflection_width, spacer_region_width]  # Layer widths in meters
        self.layer_permittivities = np.array([1.0, 3.46, 12.0, 3.46, 1.0])
        self.layer_permeabilities = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        self.boundary_refractive_index = 1.0
        
        super().__init__(max_frequency)


class BraggGrating(Device):
    def __init__(self, max_frequency):
        alternating_layer_1_width = 163e-9
        alternating_layer_2_width = 122e-9
        spacer_region = 100e-9
        n1 = 1.5  # SiN
        n2 = 2.0  # SiO2
        er1 = n1**2  # SiO2 is non-magnetic
        er2 = n2**2  # SiN is non-magnetic
        num_periods = 15

        self.layer_widths = [spacer_region] + [y for x in range(num_periods) for y in [alternating_layer_2_width, alternating_layer_1_width]] + [spacer_region] # Layer widths in meters
        self.layer_permittivities = np.array([1.0] + [y for x in range(num_periods) for y in [er2, er1]] + [1.0])
        self.layer_permeabilities = np.array([1.0] + [y for x in range(num_periods) for y in [1.0, 1.0]] + [1.0])
        self.boundary_refractive_index = 1.0
        
        super().__init__(max_frequency)