"""
Greenberg-Hastings Excitable Media Engine

A discrete-state cellular automaton modeling excitable media:
- Resting cells (state 0) become excited if enough neighbors are excited
- Excited cells (state 1) enter a refractory period
- Refractory cells cycle back to resting

Produces spiral waves, target patterns, and turbulent dynamics
similar to cardiac tissue, chemical reactions (BZ), and neural networks.
"""

import numpy as np
from .engine_base import CAEngine


def _count_moore(grid):
    """Count Moore neighborhood (8 neighbors) using np.roll."""
    n = np.zeros_like(grid, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            n += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    return n


def _count_vonneumann(grid):
    """Count Von Neumann neighborhood (4 neighbors) using np.roll."""
    return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))


class Excitable(CAEngine):

    engine_name = "excitable"
    engine_label = "Excitable Media"

    def __init__(self, size=512, num_states=8, threshold=2,
                 neighborhood="moore"):
        """
        Args:
            size: Grid dimension
            num_states: Total states (0=resting, 1=excited, 2..N-1=refractory)
            threshold: Minimum excited neighbors to trigger excitation
            neighborhood: "moore" (8) or "vonneumann" (4)
        """
        super().__init__(size)
        self.num_states = num_states
        self.threshold = threshold
        self.neighborhood = neighborhood

        # Integer state grid
        self.state = np.zeros((size, size), dtype=np.int32)

    def _count_neighbors(self, grid):
        if self.neighborhood == "vonneumann":
            return _count_vonneumann(grid)
        return _count_moore(grid)

    def step(self):
        """Advance one generation."""
        # Count excited (state==1) neighbors
        excited_mask = (self.state == 1).astype(np.float64)
        excited_neighbors = self._count_neighbors(excited_mask)
        excited_neighbors = np.round(excited_neighbors).astype(np.int32)

        new_state = np.zeros_like(self.state)

        # Resting (0) -> Excited (1) if >= threshold excited neighbors
        resting = self.state == 0
        new_state[resting & (excited_neighbors >= self.threshold)] = 1

        # Excited (1) -> Refractory (2)
        new_state[self.state == 1] = 2

        # Refractory (2..N-2) -> next refractory state
        for s in range(2, self.num_states - 1):
            new_state[self.state == s] = s + 1

        # Final refractory (N-1) -> Resting (0) (already 0 in new_state)

        self.state = new_state
        self._update_display()

        self.generation += 1
        return self.world

    def _update_display(self):
        """Map integer states to [0, 1] display values."""
        self.world[:] = 0.0
        self.world[self.state == 1] = 1.0
        if self.num_states > 2:
            for s in range(2, self.num_states):
                mask = self.state == s
                # Fade from bright to dim through refractory period
                self.world[mask] = 1.0 - (s - 1) / (self.num_states - 1)

    def apply_feedback(self, feedback):
        """Probabilistic excitation in feedback regions."""
        # Where feedback is strong, randomly excite resting cells
        prob = np.clip(feedback * 10.0, 0.0, 1.0)
        excite_mask = (self.state == 0) & (np.random.random(self.state.shape) < prob)
        self.state[excite_mask] = 1
        self.world[excite_mask] = 1.0

    def set_params(self, num_states=None, threshold=None, neighborhood=None,
                   **_kw):
        if num_states is not None:
            self.num_states = int(num_states)
            self.state = np.clip(self.state, 0, self.num_states - 1)
        if threshold is not None:
            self.threshold = int(threshold)
        if neighborhood is not None:
            self.neighborhood = neighborhood

    def get_params(self):
        return {
            "num_states": self.num_states,
            "threshold": self.threshold,
            "neighborhood": self.neighborhood,
        }

    def seed(self, seed_type="random", **kwargs):
        density = kwargs.get("density", 0.15)
        if seed_type == "random":
            self._seed_random(density)
        elif seed_type == "center_burst":
            self._seed_center_burst()
        elif seed_type == "sparse":
            self._seed_random(0.05)
        else:
            self._seed_random(density)

    def _seed_random(self, density=0.15):
        """Scatter random excited cells."""
        self.state[:] = 0
        excited = np.random.random((self.size, self.size)) < density
        self.state[excited] = 1
        self._update_display()
        self.generation = 0

    def _seed_center_burst(self):
        """Seed a dense excited patch in the center."""
        self.state[:] = 0
        r = self.size // 8
        cy, cx = self.size // 2, self.size // 2
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < r
        self.state[mask] = 1
        # Add some noise around the edge
        edge = (dist >= r) & (dist < r * 1.5)
        self.state[edge] = np.where(
            np.random.random(edge.sum()) < 0.3, 1, 0
        )
        self._update_display()
        self.generation = 0

    def add_blob(self, cx, cy, radius=15, value=0.8):
        """Paint excited cells."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        self.state[mask] = 1
        self.world[mask] = 1.0

    def remove_blob(self, cx, cy, radius=15):
        """Set cells to resting."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        self.state[mask] = 0
        self.world[mask] = 0.0

    def clear(self):
        self.state[:] = 0
        self.world[:] = 0
        self.generation = 0

    @property
    def stats(self):
        excited = int((self.state == 1).sum())
        refractory = int((self.state >= 2).sum())
        total = self.size * self.size
        return {
            "generation": self.generation,
            "mass": float(excited + refractory),
            "mean": float(self.world.mean()),
            "max": float(self.world.max()),
            "alive_pct": (excited + refractory) / total * 100,
        }

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "num_states", "label": "States", "section": "EXCITABLE MEDIA",
             "min": 3, "max": 20, "default": 8, "fmt": ".0f", "step": 1},
            {"key": "threshold", "label": "Threshold", "section": "EXCITABLE MEDIA",
             "min": 1, "max": 5, "default": 2, "fmt": ".0f", "step": 1},
        ]
