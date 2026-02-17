"""
Gray-Scott Reaction-Diffusion Engine

Two chemical species (U, V) react and diffuse on a 2D grid:
  U + 2V -> 3V  (autocatalytic reaction)
  U is continuously fed in, V is continuously removed.

Equations:
  dU/dt = Du * laplacian(U) - U*V^2 + F*(1-U)
  dV/dt = Dv * laplacian(V) + U*V^2 - (F+k)*V

Spatial containment via radial feed mask keeps patterns centered:
  - Center (within 40% radius): normal feed rate
  - Edge ramp (40%->70%): feed increases to 5x (kills V, restores U=1)
  - Beyond 70%: 5x feed — hard boundary

Flow field advection is handled externally by the viewer (universal
across all engine types).

References:
  Pearson, "Complex Patterns in a Simple System" (1993)
  Karl Sims, RD Tool (karlsims.com/rdtool.html)
"""

import numpy as np
from .engine_base import CAEngine


class GrayScott(CAEngine):

    engine_name = "gray_scott"
    engine_label = "Gray-Scott"

    def __init__(self, size=512, feed=0.055, kill=0.062,
                 Du=0.2097, Dv=0.105):
        super().__init__(size)
        self.feed = feed
        self.kill = kill
        self.Du = Du
        self.Dv = Dv

        # Two chemical species (float32 for 2x bandwidth vs float64)
        self.U = np.ones((size, size), dtype=np.float32)
        self.V = np.zeros((size, size), dtype=np.float32)

        # Pre-allocate work buffers to avoid per-frame allocation
        self._padded = np.zeros((size + 2, size + 2), dtype=np.float32)
        self._lap_U = np.empty((size, size), dtype=np.float32)
        self._lap_V = np.empty((size, size), dtype=np.float32)
        self._uvv = np.empty((size, size), dtype=np.float32)
        self._tmp = np.empty((size, size), dtype=np.float32)

        # Build radial feed mask for spatial containment
        self._feed_mask = self._build_feed_mask(size)
        # Cache feed_spatial and fk (rebuilt when feed/kill change)
        self._feed_spatial = self.feed * self._feed_mask
        self._fk = self._feed_spatial + self.kill

    def _build_feed_mask(self, size):
        """Build radial feed-rate multiplier mask.

        Center (within 40% radius): 1.0 (use preset F)
        Edge ramp (40%->70%): smoothly increases to 5.0
        Beyond 70%: 5.0 (kills V rapidly, restores U=1)
        """
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center

        # Ramp zone: 0.40 -> 0.70
        ramp = np.clip((dist - 0.40) / 0.30, 0.0, 1.0)
        # Smooth ramp from 1.0 to 5.0
        mask = (1.0 + ramp * 4.0).astype(np.float32)
        return mask

    def _laplacian(self, field, out):
        """Fast 9-point laplacian into pre-allocated output buffer.

        Uses pad+slice (one copy) instead of 8 np.roll calls.
        Weighted 9-point stencil for better isotropy.
        """
        p = self._padded
        p[1:-1, 1:-1] = field
        p[0, 1:-1] = field[-1, :]
        p[-1, 1:-1] = field[0, :]
        p[1:-1, 0] = field[:, -1]
        p[1:-1, -1] = field[:, 0]
        p[0, 0] = field[-1, -1]
        p[0, -1] = field[-1, 0]
        p[-1, 0] = field[0, -1]
        p[-1, -1] = field[0, 0]

        # Cardinal (0.2) + diagonal (0.05) - center (1.0)
        np.add(p[:-2, 1:-1], p[2:, 1:-1], out=out)
        out += p[1:-1, :-2]
        out += p[1:-1, 2:]
        out *= 0.2
        # Diagonals into _tmp, then add
        np.add(p[:-2, :-2], p[:-2, 2:], out=self._tmp)
        self._tmp += p[2:, :-2]
        self._tmp += p[2:, 2:]
        self._tmp *= 0.05
        out += self._tmp
        out -= field

    def step(self):
        """Advance one time step (2 substeps).

        Du=0.21, Dv=0.105 are 5x smaller than standard (1.0/0.5),
        so 2 substeps (dt_eff=0.5) is stable: D*dt*4 = 0.42 < 0.5.
        """
        fs = self._feed_spatial
        fk = self._fk
        Du = np.float32(self.Du)
        Dv = np.float32(self.Dv)

        for _ in range(2):
            self._laplacian(self.U, self._lap_U)
            self._laplacian(self.V, self._lap_V)

            # uvv = U * V * V
            np.multiply(self.V, self.V, out=self._uvv)
            self._uvv *= self.U

            # dU = Du*lap_U - uvv + feed*(1-U)
            self._lap_U *= Du
            self._lap_U -= self._uvv
            np.subtract(1.0, self.U, out=self._tmp)
            self._tmp *= fs
            self._lap_U += self._tmp
            self.U += self._lap_U

            # dV = Dv*lap_V + uvv - (feed+kill)*V
            self._lap_V *= Dv
            self._lap_V += self._uvv
            np.multiply(fk, self.V, out=self._tmp)
            self._lap_V -= self._tmp
            self.V += self._lap_V

            np.clip(self.U, 0.0, 1.0, out=self.U)
            np.clip(self.V, 0.0, 1.0, out=self.V)

        # Display: V concentration
        self.world = self.V
        self.generation += 1
        return self.world

    def apply_feedback(self, feedback):
        """Seed V in feedback regions, consuming U."""
        self.V = np.clip(self.V + feedback * 0.5, 0.0, 1.0)
        self.U = np.clip(self.U - feedback * 0.25, 0.0, 1.0)
        self.world = self.V

    def set_params(self, feed=None, kill=None, Du=None, Dv=None, **kwargs):
        rebuild = False
        if feed is not None:
            self.feed = feed
            rebuild = True
        if kill is not None:
            self.kill = kill
            rebuild = True
        if Du is not None:
            self.Du = Du
        if Dv is not None:
            self.Dv = Dv
        if rebuild:
            np.multiply(self._feed_mask, self.feed, out=self._feed_spatial)
            np.add(self._feed_spatial, self.kill, out=self._fk)

    def get_params(self):
        return {
            "feed": self.feed,
            "kill": self.kill,
            "Du": self.Du,
            "Dv": self.Dv,
        }

    def seed(self, seed_type="random", **kwargs):
        if seed_type == "center":
            self._seed_center()
        elif seed_type == "multi":
            self._seed_multi()
        elif seed_type == "dense":
            self._seed_dense()
        elif seed_type == "random":
            self._seed_scattered()
        else:
            self._seed_center()

    def _seed_center(self):
        """Seed a gaussian blob of V in the center."""
        self.U[:] = 1.0
        self.V[:] = 0.0
        r = self.size // 6
        center = self.size / 2.0
        Y, X = np.ogrid[:self.size, :self.size]
        dist_sq = (X - center) ** 2 + (Y - center) ** 2
        sigma_sq = r * r
        blob = 0.25 * np.exp(-dist_sq / (2.0 * sigma_sq))
        self.V = blob.astype(np.float32)
        self.U = np.clip(1.0 - blob * 2.0, 0.0, 1.0).astype(np.float32)
        # Add noise to break symmetry
        self.U += np.random.random((self.size, self.size)) * 0.02
        self.V += np.random.random((self.size, self.size)) * 0.01
        np.clip(self.U, 0, 1, out=self.U)
        np.clip(self.V, 0, 1, out=self.V)
        self.world = self.V.copy()
        self.generation = 0

    def _seed_dense(self):
        """Seed a dense noisy disk in the center.

        Fills the center 35% radius with random V concentration, giving the
        RD equations enough material to organize into patterns within ~300
        steps instead of requiring 2000+ steps from small blobs.
        """
        self.U[:] = 1.0
        self.V[:] = 0.0
        center = self.size / 2.0
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center

        # Dense noisy disk with soft gaussian edge
        disk = np.exp(-0.5 * (dist / 0.30) ** 4).astype(np.float32)
        noise = np.random.random((self.size, self.size)).astype(np.float32)
        self.V = disk * noise * 0.25
        self.U = np.clip(1.0 - self.V * 2.0, 0.0, 1.0).astype(np.float32)
        # Fine noise to break symmetry everywhere
        self.U += np.random.random((self.size, self.size)).astype(np.float32) * 0.02
        self.V += np.random.random((self.size, self.size)).astype(np.float32) * 0.005
        np.clip(self.U, 0, 1, out=self.U)
        np.clip(self.V, 0, 1, out=self.V)
        self.world = self.V.copy()
        self.generation = 0

    def _seed_multi(self):
        """Seed multiple gaussian blobs of V, center-biased placement."""
        self.U[:] = 1.0
        self.V[:] = 0.0
        r = self.size // 20
        center = self.size / 2.0
        Y, X = np.ogrid[:self.size, :self.size]
        sigma_sq = r * r
        for _ in range(6):
            # Center-biased: gaussian distribution around center
            cy = int(np.clip(np.random.normal(center, self.size * 0.15),
                             r + 10, self.size - r - 10))
            cx = int(np.clip(np.random.normal(center, self.size * 0.15),
                             r + 10, self.size - r - 10))
            dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
            blob = 0.25 * np.exp(-dist_sq / (2.0 * sigma_sq))
            self.V += blob.astype(np.float32)
            self.U -= (blob * 2.0).astype(np.float32)
        np.clip(self.U, 0, 1, out=self.U)
        np.clip(self.V, 0, 1, out=self.V)
        # Add noise to break symmetry
        self.U += np.random.random((self.size, self.size)) * 0.02
        self.V += np.random.random((self.size, self.size)) * 0.01
        np.clip(self.U, 0, 1, out=self.U)
        np.clip(self.V, 0, 1, out=self.V)
        self.world = self.V.copy()
        self.generation = 0

    def _seed_scattered(self):
        """Seed random small dots of V."""
        self.U[:] = 1.0
        self.V[:] = 0.0
        n_dots = 20
        r = max(3, self.size // 80)
        margin = r + 5
        Y, X = np.ogrid[:self.size, :self.size]
        for _ in range(n_dots):
            cy = np.random.randint(margin, self.size - margin)
            cx = np.random.randint(margin, self.size - margin)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask = dist < r
            self.U[mask] = 0.50
            self.V[mask] = 0.25
        self.U += np.random.random((self.size, self.size)) * 0.01
        self.world = self.V.copy()
        self.generation = 0

    def add_blob(self, cx, cy, radius=15, value=0.8):
        """Seed V at mouse position."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        influence = np.clip(1.0 - dist / radius, 0, 1) ** 2
        self.V = np.clip(self.V + influence * 0.25, 0, 1)
        self.U = np.clip(self.U - influence * 0.25, 0, 1)
        self.world = self.V

    def remove_blob(self, cx, cy, radius=15):
        """Remove V at mouse position, restore U."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        influence = np.clip(1.0 - dist / radius, 0, 1) ** 2
        self.V = np.clip(self.V - influence * 0.5, 0, 1)
        self.U = np.clip(self.U + influence * 0.25, 0, 1)
        self.world = self.V

    def clear(self):
        self.U[:] = 1.0
        self.V[:] = 0.0
        self.world[:] = 0.0
        self.generation = 0

    @property
    def stats(self):
        return {
            "generation": self.generation,
            "mass": float(self.V.sum()),
            "mean": float(self.V.mean()),
            "max": float(self.V.max()),
            "alive_pct": float((self.V > 0.01).sum()) / self.V.size * 100,
        }

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "feed", "label": "Feed (F)", "section": "REACTION",
             "min": 0.01, "max": 0.08, "default": 0.055, "fmt": ".4f"},
            {"key": "kill", "label": "Kill (k)", "section": "REACTION",
             "min": 0.04, "max": 0.07, "default": 0.062, "fmt": ".4f"},
            {"key": "Du", "label": "Diffuse U", "section": "DIFFUSION",
             "min": 0.10, "max": 0.30, "default": 0.2097, "fmt": ".4f"},
            {"key": "Dv", "label": "Diffuse V", "section": "DIFFUSION",
             "min": 0.03, "max": 0.15, "default": 0.105, "fmt": ".4f"},
        ]
