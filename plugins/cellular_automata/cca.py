"""
Cyclic Cellular Automata (CCA) Engine

Cells hold one of N discrete states (0..N-1) and cycle: 0→1→2→...→N-1→0.
A cell advances to the next state when at least T neighbors already hold
that next state. This creates perpetual spiral waves and turbulence —
mathematically proven to never reach equilibrium.

Parameters:
  range_r:    Neighborhood radius (1=Moore 3x3, 2=5x5, etc.)
  threshold:  Min neighbors in next state to trigger advancement
  num_states: Number of cyclic states (more = finer spirals)

Reference: Griffeath & Gravner, "Cyclic Cellular Automata in Two Dimensions"
"""

import numpy as np
from .engine_base import CAEngine


class CCA(CAEngine):

    engine_name = "cca"
    engine_label = "Cyclic CA"

    def __init__(self, size=512, range_r=1, threshold=1, num_states=14):
        super().__init__(size)
        self.range_r = range_r
        self.threshold = threshold
        self.num_states = num_states

        # Integer grid for actual CCA state
        self.grid = np.random.randint(0, num_states, (size, size), dtype=np.int16)
        # Float world for rendering pipeline
        self.world = self.grid.astype(np.float32) / max(1, num_states - 1)

        # Pre-allocate padded buffer
        self._padded = np.zeros(
            (size + 2 * range_r, size + 2 * range_r), dtype=np.int16
        )
        # Pre-allocate count buffer
        self._count = np.zeros((size, size), dtype=np.int32)

    def step(self):
        """Advance one CCA step.

        For each cell: count neighbors holding (state+1)%N.
        If count >= threshold, advance to next state.
        Uses pad+slice for fast neighbor access.
        """
        g = self.grid
        r = self.range_r
        ns = self.num_states
        s = self.size

        # Next state each cell needs from neighbors
        next_s = (g + 1) % ns

        # Pad grid with wrapping
        p = self._padded
        p[r:r + s, r:r + s] = g
        # Top/bottom
        p[:r, r:r + s] = g[-r:, :]
        p[r + s:, r:r + s] = g[:r, :]
        # Left/right
        p[r:r + s, :r] = g[:, -r:]
        p[r:r + s, r + s:] = g[:, :r]
        # Corners
        p[:r, :r] = g[-r:, -r:]
        p[:r, r + s:] = g[-r:, :r]
        p[r + s:, :r] = g[:r, -r:]
        p[r + s:, r + s:] = g[:r, :r]

        # Count neighbors matching next_state
        self._count[:] = 0
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = p[r + dy:r + dy + s, r + dx:r + dx + s]
                self._count += (neighbor == next_s)

        # Advance cells where count >= threshold
        advance = self._count >= self.threshold
        g[advance] = next_s[advance]

        # Update float world for rendering
        self.world = g.astype(np.float32) / max(1, ns - 1)
        self.generation += 1
        return self.world

    def set_params(self, range_r=None, threshold=None, num_states=None, **_kw):
        rebuild = False
        if range_r is not None and range_r != self.range_r:
            self.range_r = int(range_r)
            rebuild = True
        if threshold is not None:
            self.threshold = int(threshold)
        if num_states is not None and num_states != self.num_states:
            old_ns = self.num_states
            self.num_states = int(num_states)
            # Remap grid states to new range
            self.grid = (self.grid * self.num_states // old_ns).astype(np.int16)
        if rebuild:
            r = self.range_r
            s = self.size
            self._padded = np.zeros((s + 2 * r, s + 2 * r), dtype=np.int16)

    def get_params(self):
        return {
            "range_r": self.range_r,
            "threshold": self.threshold,
            "num_states": self.num_states,
        }

    def seed(self, seed_type="random", **kwargs):
        if seed_type == "random":
            self.grid = np.random.randint(
                0, self.num_states, (self.size, self.size), dtype=np.int16
            )
        elif seed_type == "center":
            # Random center blob, zero outside
            self.grid[:] = 0
            r = self.size // 3
            center = self.size // 2
            Y, X = np.ogrid[:self.size, :self.size]
            dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
            mask = dist < r
            self.grid[mask] = np.random.randint(
                0, self.num_states, mask.sum(), dtype=np.int16
            )
        elif seed_type == "ring":
            # Random ring pattern
            self.grid[:] = 0
            center = self.size // 2
            Y, X = np.ogrid[:self.size, :self.size]
            dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
            inner = self.size // 6
            outer = self.size // 3
            mask = (dist > inner) & (dist < outer)
            self.grid[mask] = np.random.randint(
                0, self.num_states, mask.sum(), dtype=np.int16
            )
        self.world = self.grid.astype(np.float32) / max(1, self.num_states - 1)
        self.generation = 0

    def add_blob(self, cx, cy, radius=15, value=0.8):
        """Paint random CCA states at position."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        self.grid[mask] = np.random.randint(
            0, self.num_states, mask.sum(), dtype=np.int16
        )
        self.world = self.grid.astype(np.float32) / max(1, self.num_states - 1)

    def remove_blob(self, cx, cy, radius=15):
        """Reset cells to state 0 at position."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        self.grid[mask] = 0
        self.world = self.grid.astype(np.float32) / max(1, self.num_states - 1)

    def clear(self):
        self.grid[:] = 0
        self.world[:] = 0.0
        self.generation = 0

    @property
    def stats(self):
        return {
            "generation": self.generation,
            "mass": float(self.grid.sum()),
            "mean": float(self.world.mean()),
            "max": float(self.world.max()),
            "alive_pct": float((self.grid > 0).sum()) / self.grid.size * 100,
        }

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "range_r", "label": "Range", "section": "RULE",
             "min": 1, "max": 5, "default": 1, "fmt": ".0f", "step": 1},
            {"key": "threshold", "label": "Threshold", "section": "RULE",
             "min": 1, "max": 10, "default": 1, "fmt": ".0f", "step": 1},
            {"key": "num_states", "label": "States", "section": "RULE",
             "min": 3, "max": 24, "default": 14, "fmt": ".0f", "step": 1},
        ]
