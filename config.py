import numpy as np
import device as dv
import matplotlib.pyplot as plt
import utils

c = 299792458  # Speed of light in m/s


class ProblemSetup:
    device = None
    max_frequency = 0
    figure = None
    axes = None
    rectangles = None
    time_grid = []

    def __init__(self, device_name):
        if device_name == "Slab":
            self.max_frequency = 1e9  # Hz
            self.device = dv.Slab(self.max_frequency)
        elif device_name == "AntiReflectionLayer":
            self.max_frequency = 5e9  # Hz
            self.device = dv.AntiReflectionLayer(self.max_frequency)
            
        # Initialize plot
        self.figure, self.axes = plt.subplots(nrows=2, ncols=1)
        self.rectangles = utils.create_layer_shadings(self.device)

        self.compute_time_grid()

    def compute_time_grid(self):
        # Compute time step
        nbc = self.device.boundary_refractive_index
        dz = self.device.grid_resolution
        self.time_grid = nbc*dz/(2*c)
