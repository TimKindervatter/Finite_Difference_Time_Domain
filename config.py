import numpy as np
import device as dv
import matplotlib.pyplot as plt
import utils
import fourier_transform

c = 299792458  # Speed of light in m/s


class ProblemInstance:
    device = None
    max_frequency = None
    main_figure = None
    main_axes = None
    source_axes = None
    rectangles = None
    time_grid = None
    time_step = None
    num_frequencies = None
    fourier_transform_manager = None

    def __init__(self, device_name):
        if device_name == "FreeSpace":
            self.max_frequency = 1e9  # Hz
            self.device = dv.FreeSpace(self.max_frequency)
            self.num_frequencies = 100
            self.plot_update_interval = 20
        elif device_name == "Slab":
            self.max_frequency = 1e9  # Hz
            self.device = dv.Slab(self.max_frequency)
            self.num_frequencies = 100
            self.plot_update_interval = 20
        elif device_name == "AntiReflectionLayer":
            self.max_frequency = 5e9  # Hz
            self.device = dv.AntiReflectionLayer(self.max_frequency)
            self.num_frequencies = 500
            self.plot_update_interval = 20
        elif device_name == "BraggGrating":
            self.max_frequency = 1.5e15 # Hz
            self.device = dv.BraggGrating(self.max_frequency)
            self.num_frequencies = 500
            self.plot_update_interval = 20
        else:
            raise Exception("Device not recognized.")
            
        # Initialize plot
        self.main_figure, self.main_axes = plt.subplots(nrows=2, ncols=1)
        self.rectangles = utils.create_layer_shadings(self.device)

        self.time_step = self.compute_time_step()
        self.fourier_transform_manager = fourier_transform.FourierTransform(self.num_frequencies, self.max_frequency, self.time_step)

    def compute_time_step(self):
        # Compute time step
        nbc = self.device.boundary_refractive_index
        dz = self.device.grid_resolution
        time_step = nbc*dz/(2*c)

        return time_step
