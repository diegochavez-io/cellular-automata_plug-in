"""
Continuous Multi-Neighborhood Cellular Automata (MNCA)

MNCA uses multiple concentric ring neighborhoods, each with independent
threshold rules. When a ring's average is within a threshold range, the
cell increments; outside, it decrements. Multiple rings create complex
interactions that produce self-organizing solitons, mitosis, and
emergent pursuit behaviors.

Key properties:
- Continuous [0,1] float states -> no banding or pixelation
- Multiple FFT ring kernels for multi-scale neighborhood sensing
- Threshold rules per ring: increment if avg in [low, high], else decrement
- Self-contained solitons that travel and interact autonomously
- Never reaches equilibrium (multi-ring dynamics prevent convergence)

Parameters:
    rings:  List of (inner_r, outer_r) defining ring neighborhoods
    rules:  List of rule lists per ring: [{"low": float, "high": float}]
    delta:  Base increment/decrement magnitude

Reference: Slackermanz's MNCA explorations, Softology's MNCA guide
"""

import numpy as np
from .engine_base import CAEngine


class MNCA(CAEngine):

    engine_name = "mnca"
    engine_label = "MNCA"

    def __init__(self, size=512, rings=None, rules=None, delta=0.05):
        super().__init__(size)

        # Default: two rings with classic soliton-producing rules
        if rings is None:
            rings = [(0, 5), (8, 15)]
        if rules is None:
            rules = [
                [{"low": 0.20, "high": 0.26}],   # ring 0: narrow growth band
                [{"low": 0.26, "high": 0.46}],    # ring 1: wider stability band
            ]

        self.rings = rings
        self.rules = rules
        self.delta = delta

        self._build_kernels()

    def _build_kernels(self):
        """Build antialiased ring kernels and pre-compute their FFTs."""
        size = self.size
        self._kernel_ffts = []

        for inner_r, outer_r in self.rings:
            # Distance matrix
            mid = outer_r + 4
            y, x = np.ogrid[-mid:mid+1, -mid:mid+1]
            D = np.sqrt(x*x + y*y)

            # Antialiased ring: smooth logistic edges
            if inner_r > 0:
                inner_edge = 1.0 / (1.0 + np.exp(-(D - inner_r) * 5.0))
            else:
                inner_edge = np.ones_like(D)
            outer_edge = 1.0 / (1.0 + np.exp((D - outer_r) * 5.0))
            kernel = inner_edge * outer_edge

            # Normalize to compute average
            total = kernel.sum()
            if total > 0:
                kernel /= total

            # Pad to world size and FFT
            kh, kw = kernel.shape
            padded = np.zeros((size, size), dtype=np.float64)
            padded[:kh, :kw] = kernel
            padded = np.roll(np.roll(padded, -kh // 2, axis=0), -kw // 2, axis=1)
            self._kernel_ffts.append(np.fft.rfft2(padded))

    def step(self):
        """Advance one MNCA step.

        For each ring: compute neighborhood average, check against thresholds.
        Accumulate increments/decrements from all rings, then apply.
        """
        world_fft = np.fft.rfft2(self.world)

        # Accumulate delta from all rings
        total_delta = np.zeros((self.size, self.size), dtype=np.float64)

        for i, kfft in enumerate(self._kernel_ffts):
            # Compute neighborhood average for this ring
            avg = np.fft.irfft2(world_fft * kfft, s=(self.size, self.size))

            # Apply rules for this ring
            ring_rules = self.rules[i] if i < len(self.rules) else []
            ring_active = np.zeros((self.size, self.size), dtype=bool)

            for rule in ring_rules:
                low = rule["low"]
                high = rule["high"]
                in_range = (avg >= low) & (avg <= high)
                ring_active |= in_range

            # Increment where any rule matches, decrement where none match
            total_delta += np.where(ring_active, self.delta, -self.delta)

        self.world = np.clip(self.world + total_delta, 0.0, 1.0)
        self.generation += 1
        return self.world

    def set_params(self, rings=None, rules=None, delta=None,
                   inner_r0=None, outer_r0=None, **_kw):
        rebuild = False

        if delta is not None:
            self.delta = delta
        if rings is not None:
            self.rings = rings
            rebuild = True
        if rules is not None:
            self.rules = rules
            rebuild = True

        # Slider-driven ring 0 adjustment
        if inner_r0 is not None or outer_r0 is not None:
            if len(self.rings) > 0:
                old = list(self.rings[0])
                if inner_r0 is not None:
                    old[0] = int(inner_r0)
                if outer_r0 is not None:
                    old[1] = int(outer_r0)
                self.rings[0] = tuple(old)
                rebuild = True

        if rebuild:
            self._build_kernels()

    def get_params(self):
        result = {
            "delta": self.delta,
            "rings": self.rings,
            "rules": self.rules,
        }
        if len(self.rings) > 0:
            result["inner_r0"] = self.rings[0][0]
            result["outer_r0"] = self.rings[0][1]
        return result

    def seed(self, seed_type="random", **kwargs):
        if seed_type == "blobs":
            self.seed_multiple_blobs(**kwargs)
        elif seed_type == "ring":
            self.seed_ring(**kwargs)
        elif seed_type == "sparse":
            self.seed_sparse(**kwargs)
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
        scatter = self.size * 0.15
        Y, X = np.ogrid[:self.size, :self.size]
        for _ in range(n_blobs):
            cy = int(center + np.random.randn() * scatter)
            cx = int(center + np.random.randn() * scatter)
            cy = max(blob_radius, min(self.size - blob_radius, cy))
            cx = max(blob_radius, min(self.size - blob_radius, cx))
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
            blob = np.exp(-0.5 * (dist / (blob_radius * 0.5)) ** 2) * density
            noise = np.random.random((self.size, self.size)) * 0.4 + 0.6
            self.world += blob * noise
        np.clip(self.world, 0, 1, out=self.world)
        self.generation = 0

    def seed_ring(self, radius=None, thickness=None, value=0.8):
        """Seed with a smooth ring pattern."""
        if radius is None:
            radius = self.size // 5
        if thickness is None:
            thickness = max(10, self.size // 50)
        self.world[:] = 0
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
        ring = np.exp(-0.5 * ((dist - radius) / thickness) ** 2) * value
        noise = np.random.random((self.size, self.size)) * 0.2 + 0.8
        self.world = ring * noise
        np.clip(self.world, 0, 1, out=self.world)
        self.generation = 0

    def seed_sparse(self, n_dots=12, dot_radius=None, **kwargs):
        """Seed with small scattered dots for soliton formation."""
        if dot_radius is None:
            dot_radius = self.size // 20
        self.world[:] = 0
        center = self.size // 2
        scatter = self.size * 0.25
        Y, X = np.ogrid[:self.size, :self.size]
        for _ in range(n_dots):
            cy = int(center + np.random.randn() * scatter)
            cx = int(center + np.random.randn() * scatter)
            cy = max(dot_radius, min(self.size - dot_radius, cy))
            cx = max(dot_radius, min(self.size - dot_radius, cx))
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float64)
            dot = np.exp(-0.5 * (dist / (dot_radius * 0.4)) ** 2) * 0.7
            self.world += dot
        np.clip(self.world, 0, 1, out=self.world)
        self.generation = 0

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "delta", "label": "Delta", "section": "RULES",
             "min": 0.005, "max": 0.15, "default": 0.05, "fmt": ".3f"},
            {"key": "inner_r0", "label": "Ring 0 inner", "section": "NEIGHBORHOODS",
             "min": 0, "max": 10, "default": 0, "fmt": ".0f", "step": 1},
            {"key": "outer_r0", "label": "Ring 0 outer", "section": "NEIGHBORHOODS",
             "min": 2, "max": 25, "default": 5, "fmt": ".0f", "step": 1},
        ]
