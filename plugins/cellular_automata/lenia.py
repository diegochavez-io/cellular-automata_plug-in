"""
Lenia - Continuous Cellular Automaton Engine

A continuous generalization of Conway's Game of Life where:
- States are continuous [0, 1] instead of binary
- Neighborhoods use smooth kernel functions instead of discrete counts
- Growth/decay is governed by a Gaussian growth function
- Time steps are fractional for smooth evolution

Reference: Bert Chan, "Lenia - Biology of Artificial Life" (2020)
"""

import numpy as np
from .engine_base import CAEngine


class Lenia(CAEngine):

    engine_name = "lenia"
    engine_label = "Lenia"

    def __init__(self, size=512, R=13, T=10, mu=0.15, sigma=0.017,
                 kernel_peaks=None, kernel_widths=None):
        """
        Args:
            size: Grid dimension (size x size)
            R: Kernel radius in cells
            T: Time resolution (dt = 1/T, higher = smoother/slower)
            mu: Growth function center (neighborhood density that promotes growth)
            sigma: Growth function width (tolerance around mu)
            kernel_peaks: List of radial peak positions for kernel rings [0-1]
            kernel_widths: List of widths for each kernel ring
        """
        super().__init__(size)
        self.R = R
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.dt = 1.0 / T
        self.kernel_peaks = kernel_peaks or [0.5]
        self.kernel_widths = kernel_widths or [0.15]

        self._build_kernel()

    def _bell(self, x, center, width):
        """Gaussian bell curve"""
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    def _build_kernel(self):
        """Build convolution kernel and pre-compute its FFT"""
        # Distance matrix from center, normalized to [0, 1] at radius R
        mid = self.R
        y, x = np.ogrid[-mid:mid+1, -mid:mid+1]
        D = np.sqrt(x*x + y*y) / self.R

        # Build kernel as sum of ring functions
        K = np.zeros_like(D)
        for peak, width in zip(self.kernel_peaks, self.kernel_widths):
            K += self._bell(D, peak, width)

        # Zero outside unit circle
        K[D > 1] = 0

        # Normalize to sum to 1
        total = K.sum()
        if total > 0:
            K /= total

        self._kernel_raw = K

        # Pad kernel to world size and pre-compute FFT for fast convolution
        padded = np.zeros((self.size, self.size), dtype=np.float64)
        kh, kw = K.shape
        padded[:kh, :kw] = K
        # Roll so kernel center is at (0,0) for circular convolution
        padded = np.roll(np.roll(padded, -kh // 2, axis=0), -kw // 2, axis=1)
        self._kernel_fft = np.fft.rfft2(padded)

    def growth(self, U):
        """Growth mapping: maps neighborhood potential to growth rate [-1, 1]"""
        return 2.0 * self._bell(U, self.mu, self.sigma) - 1.0

    def step(self):
        """Advance one time step. Returns the world state."""
        # Convolve world with kernel using FFT (periodic boundaries)
        world_fft = np.fft.rfft2(self.world)
        U = np.fft.irfft2(world_fft * self._kernel_fft, s=(self.size, self.size))

        # Apply growth function and update
        self.world = np.clip(self.world + self.dt * self.growth(U), 0.0, 1.0)
        self.generation += 1
        return self.world

    def apply_feedback(self, feedback):
        """Add feedback directly to the continuous world state."""
        self.world = np.clip(self.world + feedback, 0.0, 1.0)

    def set_params(self, mu=None, sigma=None, T=None, R=None,
                   kernel_peaks=None, kernel_widths=None, **_kw):
        """Update parameters. Rebuilds kernel if R or kernel shape changes."""
        rebuild = False

        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        if T is not None:
            self.T = T
            self.dt = 1.0 / T
        if R is not None and R != self.R:
            self.R = R
            rebuild = True
        if kernel_peaks is not None:
            self.kernel_peaks = kernel_peaks
            rebuild = True
        if kernel_widths is not None:
            self.kernel_widths = kernel_widths
            rebuild = True

        if rebuild:
            self._build_kernel()

    def get_params(self):
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "R": self.R,
            "T": self.T,
            "kernel_peaks": self.kernel_peaks,
            "kernel_widths": self.kernel_widths,
        }

    def seed(self, seed_type="random", **kwargs):
        """Seed the world based on type string."""
        if seed_type == "blobs":
            self.seed_multiple_blobs(**kwargs)
        elif seed_type == "ring":
            self.seed_ring(**kwargs)
        elif seed_type == "dense":
            self.seed_random(density=0.8, radius=self.size // 3)
        else:
            self.seed_random(**kwargs)

    def seed_random(self, density=0.5, radius=None):
        """Seed a circular region with gaussian-falloff random values."""
        if radius is None:
            radius = self.size // 4
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # Smooth gaussian envelope instead of hard circle
        envelope = np.exp(-0.5 * (dist / (radius * 0.6)) ** 2)
        noise = np.random.random((self.size, self.size)) * density
        self.world = noise * envelope
        np.clip(self.world, 0, 1, out=self.world)
        self.generation = 0

    def seed_multiple_blobs(self, n_blobs=8, blob_radius=None, density=0.6):
        """Seed with overlapping gaussian blobs clustered near center."""
        if blob_radius is None:
            blob_radius = self.size // 10
        self.world[:] = 0
        center = self.size // 2
        # Scatter radius — blobs stay within central 40% of frame
        scatter = self.size * 0.2
        Y, X = np.ogrid[:self.size, :self.size]
        for _ in range(n_blobs):
            # Center-biased placement (gaussian scatter)
            cy = int(center + np.random.randn() * scatter)
            cx = int(center + np.random.randn() * scatter)
            cy = max(blob_radius, min(self.size - blob_radius, cy))
            cx = max(blob_radius, min(self.size - blob_radius, cx))
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
            # Smooth gaussian blob — no hard edge
            blob = np.exp(-0.5 * (dist / (blob_radius * 0.5)) ** 2) * density
            # Add noise texture within the blob
            noise = np.random.random((self.size, self.size)) * 0.4 + 0.6
            self.world += blob * noise
        np.clip(self.world, 0, 1, out=self.world)
        self.generation = 0

    def seed_ring(self, radius=None, thickness=None, value=0.8):
        """Seed with a smooth ring pattern - great for producing spiral waves."""
        if radius is None:
            radius = self.size // 5
        if thickness is None:
            thickness = max(10, self.size // 50)
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
        # Smooth ring profile — gaussian cross-section
        ring = np.exp(-0.5 * ((dist - radius) / thickness) ** 2) * value
        # Add noise to break symmetry
        noise = np.random.random((self.size, self.size)) * 0.2 + 0.8
        self.world = ring * noise
        np.clip(self.world, 0, 1, out=self.world)
        self.generation = 0

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "mu", "label": "Growth", "section": "SHAPE",
             "min": 0.08, "max": 0.28, "default": 0.15, "fmt": ".3f"},
            {"key": "sigma", "label": "Tolerance", "section": "SHAPE",
             "min": 0.005, "max": 0.045, "default": 0.017, "fmt": ".3f"},
            {"key": "T", "label": "Smoothness", "section": "SHAPE",
             "min": 5, "max": 25, "default": 10, "fmt": ".0f", "step": 1},
        ]
