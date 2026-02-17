"""
SmoothLife - Continuous Generalization of Game of Life

SmoothLife replaces discrete cell counts with continuous integrals over
smooth disk/annulus neighborhoods, and replaces binary alive/dead states
with continuous [0,1] values. The transition function uses sigmoid intervals
for birth and survival thresholds.

Key properties:
- Continuous states and neighborhoods -> inherently smooth, no pixelation
- Never reaches static equilibrium (continuous dynamics prevent lock-in)
- Two FFT kernels: inner disk (filling) and outer annulus (neighborhood)
- Sigmoid-based birth/death transitions with tunable sharpness

Parameters:
    ri:      Inner disk radius (for filling measurement)
    ra:      Outer annulus radius (ra = 3*ri by default)
    b1, b2:  Birth interval [b1, b2] (outer density that triggers birth)
    d1, d2:  Survival interval [d1, d2] (outer density that allows survival)
    alpha_n: Transition sharpness for outer neighborhood sigmoid
    alpha_m: Transition sharpness for inner filling sigmoid
    dt:      Time step size

Reference: Rafler, "Generalization of Conway's Game of Life to a
           continuous domain - SmoothLife" (2011)
"""

import numpy as np
from .engine_base import CAEngine


class SmoothLife(CAEngine):

    engine_name = "smoothlife"
    engine_label = "SmoothLife"

    def __init__(self, size=512, ri=4, ra=12,
                 b1=0.278, b2=0.365, d1=0.267, d2=0.445,
                 alpha_n=0.028, alpha_m=0.147, dt=0.1):
        super().__init__(size)
        self.ri = ri
        self.ra = ra
        self.b1 = b1
        self.b2 = b2
        self.d1 = d1
        self.d2 = d2
        self.alpha_n = alpha_n
        self.alpha_m = alpha_m
        self.dt = dt

        self._build_kernels()

    def _sigma(self, x, a, alpha):
        """Smooth step function (logistic sigmoid)."""
        return 1.0 / (1.0 + np.exp(-(x - a) / max(alpha, 1e-6)))

    def _sigma_n(self, x, a, b):
        """Smooth interval function: 1 when a < x < b, 0 outside."""
        return self._sigma(x, a, self.alpha_n) * (1.0 - self._sigma(x, b, self.alpha_n))

    def _sigma_m(self, x, y, m):
        """Interpolate between x and y based on filling m."""
        return x * (1.0 - self._sigma(m, 0.5, self.alpha_m)) + \
               y * self._sigma(m, 0.5, self.alpha_m)

    def _build_kernels(self):
        """Build inner disk and outer annulus kernels with antialiased edges.

        Uses logistic falloff at boundaries for smooth kernels (no hard edges).
        Pre-computes FFTs for fast convolution.
        """
        size = self.size
        ri, ra = self.ri, self.ra

        # Distance from center
        mid = max(ri, ra) + 4  # extra padding for falloff
        y, x = np.ogrid[-mid:mid+1, -mid:mid+1]
        D = np.sqrt(x*x + y*y)

        # Antialiased disk (inner): smooth falloff at ri boundary
        # Width of transition zone ~1 cell for smooth edge
        M_kernel = 1.0 / (1.0 + np.exp((D - ri) * 5.0))

        # Antialiased annulus (outer): ring between ri and ra
        inner_edge = 1.0 / (1.0 + np.exp(-(D - ri) * 5.0))  # 0 inside ri
        outer_edge = 1.0 / (1.0 + np.exp((D - ra) * 5.0))   # 0 outside ra
        N_kernel = inner_edge * outer_edge

        # Normalize
        m_sum = M_kernel.sum()
        n_sum = N_kernel.sum()
        if m_sum > 0:
            M_kernel /= m_sum
        if n_sum > 0:
            N_kernel /= n_sum

        # Pad to world size and compute FFT
        kh, kw = M_kernel.shape

        M_padded = np.zeros((size, size), dtype=np.float32)
        M_padded[:kh, :kw] = M_kernel
        M_padded = np.roll(np.roll(M_padded, -kh // 2, axis=0), -kw // 2, axis=1)
        self._M_fft = np.fft.rfft2(M_padded)

        N_padded = np.zeros((size, size), dtype=np.float32)
        N_padded[:kh, :kw] = N_kernel
        N_padded = np.roll(np.roll(N_padded, -kh // 2, axis=0), -kw // 2, axis=1)
        self._N_fft = np.fft.rfft2(N_padded)

    def _transition(self, n, m):
        """Compute transition function S(n, m).

        n: outer neighborhood density (annulus average)
        m: inner filling (disk average)

        Birth: when outer density is in [b1, b2]
        Survival: when outer density is in [d1, d2]
        Interpolated by inner filling m.
        """
        birth = self._sigma_n(n, self.b1, self.b2)
        survival = self._sigma_n(n, self.d1, self.d2)
        return self._sigma_m(birth, survival, m)

    def step(self):
        """Advance one SmoothLife time step."""
        world_fft = np.fft.rfft2(self.world)

        # Inner filling (disk convolution)
        m = np.fft.irfft2(world_fft * self._M_fft, s=(self.size, self.size))
        # Outer neighborhood (annulus convolution)
        n = np.fft.irfft2(world_fft * self._N_fft, s=(self.size, self.size))

        # Continuous time step toward transition state
        S = self._transition(n, m)
        self.world = np.clip(self.world + self.dt * (2.0 * S - 1.0), 0.0, 1.0)

        self.generation += 1
        return self.world

    def set_params(self, ri=None, ra=None, b1=None, b2=None,
                   d1=None, d2=None, alpha_n=None, alpha_m=None,
                   dt=None, **_kw):
        rebuild = False

        if b1 is not None:
            self.b1 = b1
        if b2 is not None:
            self.b2 = b2
        if d1 is not None:
            self.d1 = d1
        if d2 is not None:
            self.d2 = d2
        if alpha_n is not None:
            self.alpha_n = alpha_n
        if alpha_m is not None:
            self.alpha_m = alpha_m
        if dt is not None:
            self.dt = dt
        if ri is not None and ri != self.ri:
            self.ri = ri
            rebuild = True
        if ra is not None and ra != self.ra:
            self.ra = ra
            rebuild = True

        if rebuild:
            self._build_kernels()

    def get_params(self):
        return {
            "ri": self.ri,
            "ra": self.ra,
            "b1": self.b1,
            "b2": self.b2,
            "d1": self.d1,
            "d2": self.d2,
            "alpha_n": self.alpha_n,
            "alpha_m": self.alpha_m,
            "dt": self.dt,
        }

    def seed(self, seed_type="random", **kwargs):
        if seed_type == "blobs":
            self.seed_multiple_blobs(**kwargs)
        elif seed_type == "ring":
            self.seed_ring(**kwargs)
        else:
            self.seed_random(**kwargs)
        # Ensure float32 for fast FFT (seed methods may produce float64 via np.random)
        if self.world.dtype != np.float32:
            self.world = self.world.astype(np.float32)

    def seed_random(self, density=0.5, radius=None):
        """Seed a dense filled circle with near-binary values.

        SmoothLife needs high-density seeds (values near 1.0) so that
        neighborhood averages fall within the birth/survival thresholds.
        """
        if radius is None:
            radius = self.size // 5
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # Dense filled circle: values 0.7-1.0 inside, soft edge
        inner = dist < radius
        self.world[inner] = np.random.random(inner.sum()) * 0.3 + 0.7
        # Soft edge transition
        edge_width = max(3, int(self.ra * 0.5))
        edge = (dist >= radius) & (dist < radius + edge_width)
        fade = 1.0 - (dist[edge] - radius) / edge_width
        self.world[edge] = np.random.random(edge.sum()) * 0.3 * fade

        # Break symmetry with noise
        noise = np.random.randn(self.size, self.size) * 0.05
        self.world = np.clip(self.world + noise, 0.0, 1.0)
        self.generation = 0

    def seed_multiple_blobs(self, n_blobs=6, blob_radius=None, density=0.6):
        """Seed with overlapping dense blobs clustered near center."""
        if blob_radius is None:
            blob_radius = max(10, int(self.ra * 2.5))
        self.world[:] = 0
        center = self.size // 2
        scatter = self.size * 0.12
        Y, X = np.ogrid[:self.size, :self.size]
        for _ in range(n_blobs):
            cy = int(center + np.random.randn() * scatter)
            cx = int(center + np.random.randn() * scatter)
            cy = max(blob_radius, min(self.size - blob_radius, cy))
            cx = max(blob_radius, min(self.size - blob_radius, cx))
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
            # Dense blob: near-binary values
            mask = dist < blob_radius
            values = np.random.random((self.size, self.size)) * 0.25 + 0.75
            self.world = np.where(mask, np.maximum(self.world, values * density), self.world)
        # Break symmetry
        noise = np.random.randn(self.size, self.size) * 0.03
        self.world = np.clip(self.world + noise, 0.0, 1.0)
        self.generation = 0

    def seed_ring(self, radius=None, thickness=None, value=0.8):
        """Seed with a dense ring pattern."""
        if radius is None:
            radius = self.size // 5
        if thickness is None:
            thickness = max(8, int(self.ra * 1.5))
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
        # Dense ring: near-binary values in the ring band
        ring_mask = (dist > radius - thickness) & (dist < radius + thickness)
        self.world[ring_mask] = np.random.random(ring_mask.sum()) * 0.2 + value
        # Break symmetry
        noise = np.random.randn(self.size, self.size) * 0.03
        self.world = np.clip(self.world + noise, 0.0, 1.0)
        self.generation = 0

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "b1", "label": "Birth", "section": "SHAPE",
             "min": 0.18, "max": 0.38, "default": 0.278, "fmt": ".3f"},
            {"key": "b2", "label": "Spread", "section": "SHAPE",
             "min": 0.28, "max": 0.50, "default": 0.365, "fmt": ".3f"},
            {"key": "d2", "label": "Survive", "section": "SHAPE",
             "min": 0.35, "max": 0.60, "default": 0.445, "fmt": ".3f"},
        ]
