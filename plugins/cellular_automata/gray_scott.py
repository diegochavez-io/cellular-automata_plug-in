"""
Gray-Scott Reaction-Diffusion Engine

Two chemical species (U, V) react and diffuse on a 2D grid:
  U + 2V -> 3V  (autocatalytic reaction)
  U is continuously fed in, V is continuously removed.

Equations:
  dU/dt = Du * laplacian(U) - U*V^2 + F*(1-U)
  dV/dt = Dv * laplacian(V) + U*V^2 - (F+k)*V

Produces spots, stripes, labyrinthine patterns, mitosis, and more
depending on feed rate F and kill rate k.

Reference: Pearson, "Complex Patterns in a Simple System" (1993)
"""

import numpy as np
from .engine_base import CAEngine


def _laplacian(field):
    """Compute discrete Laplacian using np.roll (periodic boundaries).
    Uses the weighted 9-point stencil for better isotropy."""
    # Cardinal directions (weight 0.2)
    lap = 0.2 * (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                  np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))
    # Diagonal directions (weight 0.05)
    lap += 0.05 * (np.roll(np.roll(field, 1, axis=0), 1, axis=1) +
                    np.roll(np.roll(field, 1, axis=0), -1, axis=1) +
                    np.roll(np.roll(field, -1, axis=0), 1, axis=1) +
                    np.roll(np.roll(field, -1, axis=0), -1, axis=1))
    lap -= field
    return lap


class GrayScott(CAEngine):

    engine_name = "gray_scott"
    engine_label = "Gray-Scott"

    def __init__(self, size=512, feed=0.055, kill=0.062,
                 Du=0.2097, Dv=0.105):
        """
        Args:
            size: Grid dimension
            feed: Feed rate F (how fast U is replenished)
            kill: Kill rate k (how fast V is removed)
            Du: Diffusion coefficient for U
            Dv: Diffusion coefficient for V
        """
        super().__init__(size)
        self.feed = feed
        self.kill = kill
        self.Du = Du
        self.Dv = Dv

        # Two chemical species
        self.U = np.ones((size, size), dtype=np.float64)
        self.V = np.zeros((size, size), dtype=np.float64)

    def step(self):
        """Advance one time step (multiple substeps for stability)."""
        dt = 1.0
        for _ in range(4):
            lap_U = _laplacian(self.U)
            lap_V = _laplacian(self.V)

            uvv = self.U * self.V * self.V

            self.U += dt * (self.Du * lap_U - uvv + self.feed * (1.0 - self.U))
            self.V += dt * (self.Dv * lap_V + uvv - (self.feed + self.kill) * self.V)

            self.U = np.clip(self.U, 0.0, 1.0)
            self.V = np.clip(self.V, 0.0, 1.0)

        # Display: V concentration
        self.world = self.V
        self.generation += 1
        return self.world

    def apply_feedback(self, feedback):
        """Seed V in feedback regions, consuming U."""
        self.V = np.clip(self.V + feedback * 0.5, 0.0, 1.0)
        self.U = np.clip(self.U - feedback * 0.25, 0.0, 1.0)
        self.world = self.V

    def set_params(self, feed=None, kill=None, Du=None, Dv=None, **_kw):
        if feed is not None:
            self.feed = feed
        if kill is not None:
            self.kill = kill
        if Du is not None:
            self.Du = Du
        if Dv is not None:
            self.Dv = Dv

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
        elif seed_type == "random":
            self._seed_scattered()
        else:
            self._seed_center()

    def _seed_center(self):
        """Seed a square patch of V in the center."""
        self.U[:] = 1.0
        self.V[:] = 0.0
        r = self.size // 10
        cy, cx = self.size // 2, self.size // 2
        self.U[cy-r:cy+r, cx-r:cx+r] = 0.50
        self.V[cy-r:cy+r, cx-r:cx+r] = 0.25
        # Add noise to break symmetry
        self.U += np.random.random((self.size, self.size)) * 0.02
        self.V += np.random.random((self.size, self.size)) * 0.02
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)
        self.world = self.V.copy()
        self.generation = 0

    def _seed_multi(self):
        """Seed multiple small patches of V."""
        self.U[:] = 1.0
        self.V[:] = 0.0
        r = self.size // 20
        margin = r + 20
        for _ in range(6):
            cy = np.random.randint(margin, self.size - margin)
            cx = np.random.randint(margin, self.size - margin)
            self.U[cy-r:cy+r, cx-r:cx+r] = 0.50
            self.V[cy-r:cy+r, cx-r:cx+r] = 0.25
        self.U += np.random.random((self.size, self.size)) * 0.02
        self.V += np.random.random((self.size, self.size)) * 0.02
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)
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
