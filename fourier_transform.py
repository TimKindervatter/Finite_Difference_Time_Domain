import numpy as np
from numba import jitclass
from numba import int16, float32, double, complex128


spec = [
    ('num_frequencies', int16),
    ('time_step', float32),
    ('freq', double[:]),
    ('kernel', complex128[:]),
    ('reflected_fourier', complex128[:]),
    ('transmitted_fourier', complex128[:]),
    ('source_fourier', complex128[:]),
    ('reflectance', double[:]),
    ('transmittance', double[:]),
    ('conservation_of_energy', double[:]),
]


@jitclass(spec)
class FourierTransform:
    def __init__(self, Nfreq, max_frequency, time_step):
        # Initialize Fourier Transforms
        self.num_frequencies = Nfreq
        self.time_step = time_step
        self.freq = np.linspace(0, max_frequency, self.num_frequencies)
        self.kernel = np.exp(-1j*2*np.pi*time_step*self.freq)

        # Initialize Fourier transforms for reflected and transmitted fields
        self.reflected_fourier = np.zeros(Nfreq, dtype=np.complex128)
        self.transmitted_fourier = np.zeros(Nfreq, dtype=np.complex128)
        self.source_fourier = np.zeros(Nfreq, dtype=np.complex128)

        self.reflectance = np.zeros(Nfreq, dtype=np.double)
        self.transmittance = np.zeros(Nfreq, dtype=np.double)
        self.conservation_of_energy = np.zeros(Nfreq, dtype=np.double)

    def update_fourier_transforms(self, T, Ey, Eysrc, Nz):
        # Update Fourier transforms
        #TODO: Parallelize Fourier computation
        for f in range(self.num_frequencies):
            self.reflected_fourier[f] = self.reflected_fourier[f] + (self.kernel[f]**T)*Ey[0]
            self.transmitted_fourier[f] = self.transmitted_fourier[f] + (self.kernel[f]**T)*Ey[Nz - 1]
            self.source_fourier[f] = self.source_fourier[f] + (self.kernel[f]**T)*Eysrc[T]

        self.reflectance = np.abs(self.reflected_fourier/self.source_fourier)**2
        self.transmittance = np.abs(self.transmitted_fourier/self.source_fourier)**2
        self.conservation_of_energy = self.reflectance + self.transmittance


    def finalize_fourier_transforms(self):
        self.reflected_fourier = self.reflected_fourier*self.time_step
        self.transmitted_fourier = self.transmitted_fourier*self.time_step
        self.source_fourier = self.source_fourier*self.time_step