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


class Lenia:
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
        self.size = size
        self.R = R
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.dt = 1.0 / T
        self.kernel_peaks = kernel_peaks or [0.5]
        self.kernel_widths = kernel_widths or [0.15]

        self.world = np.zeros((size, size), dtype=np.float64)
        self._build_kernel()
        self.generation = 0

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

    def step_n(self, n):
        """Advance n steps. Returns final state."""
        for _ in range(n):
            self.step()
        return self.world

    def set_params(self, mu=None, sigma=None, T=None, R=None,
                   kernel_peaks=None, kernel_widths=None):
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

    def seed_random(self, density=0.5, radius=None):
        """Seed a circular region with random values."""
        if radius is None:
            radius = self.size // 4
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = dist < radius
        self.world[mask] = np.random.random(mask.sum()) * density
        self.generation = 0

    def seed_multiple_blobs(self, n_blobs=8, blob_radius=None, density=0.6):
        """Seed with multiple random circular blobs."""
        if blob_radius is None:
            blob_radius = self.size // 12
        self.world[:] = 0
        margin = blob_radius + 10
        Y, X = np.ogrid[:self.size, :self.size]
        for _ in range(n_blobs):
            cy = np.random.randint(margin, self.size - margin)
            cx = np.random.randint(margin, self.size - margin)
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = dist < blob_radius
            self.world[mask] = np.clip(
                self.world[mask] + np.random.random(mask.sum()) * density,
                0, 1
            )
        self.generation = 0

    def seed_ring(self, radius=None, thickness=10, value=0.8):
        """Seed with a ring pattern - great for producing spiral waves."""
        if radius is None:
            radius = self.size // 5
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = (dist > radius - thickness) & (dist < radius + thickness)
        self.world[mask] = value
        # Add some noise to break symmetry
        self.world[mask] += np.random.random(mask.sum()) * 0.2 - 0.1
        self.world = np.clip(self.world, 0, 1)
        self.generation = 0

    def add_blob(self, cx, cy, radius=15, value=0.8):
        """Add a single blob at position (cx, cy). For mouse interaction."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        # Smooth falloff
        influence = np.clip(1.0 - dist / radius, 0, 1) ** 2 * value
        self.world = np.clip(self.world + influence, 0, 1)

    def remove_blob(self, cx, cy, radius=15):
        """Remove matter at position (cx, cy). For mouse interaction."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        influence = np.clip(1.0 - dist / radius, 0, 1) ** 2
        self.world = np.clip(self.world - influence, 0, 1)

    def clear(self):
        """Clear the world."""
        self.world[:] = 0
        self.generation = 0

    @property
    def stats(self):
        """Return current world statistics."""
        return {
            "generation": self.generation,
            "mass": float(self.world.sum()),
            "mean": float(self.world.mean()),
            "max": float(self.world.max()),
            "alive_pct": float((self.world > 0.01).sum()) / self.world.size * 100,
        }
