"""
Game of Life Engine - Classic and Variant Cellular Automata

Supports arbitrary B/S (birth/survival) rule notation:
- B3/S23: Conway's Game of Life
- B36/S23: HighLife (self-replicating)
- B3678/S34678: Day & Night

Features smooth fade mode where dead cells decay gradually,
making the binary CA look beautiful with continuous colormaps.
"""

import numpy as np
from .engine_base import CAEngine


def parse_rule(rule_str):
    """Parse B/S notation like 'B3/S23' into (birth_set, survive_set)."""
    rule_str = rule_str.upper().replace(" ", "")
    parts = rule_str.split("/")
    birth = set()
    survive = set()
    for part in parts:
        if part.startswith("B"):
            birth = {int(c) for c in part[1:]}
        elif part.startswith("S"):
            survive = {int(c) for c in part[1:]}
    return birth, survive


def _count_neighbors_moore(grid):
    """Count Moore neighborhood (8 neighbors) using np.roll with periodic boundaries."""
    n = np.zeros_like(grid, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            n += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    return n


def _count_neighbors_vonneumann(grid):
    """Count Von Neumann neighborhood (4 neighbors) using np.roll."""
    return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1))


class Life(CAEngine):

    engine_name = "life"
    engine_label = "Game of Life"

    def __init__(self, size=512, rule="B3/S23", neighborhood="moore",
                 fade_rate=0.92):
        """
        Args:
            size: Grid dimension
            rule: B/S rule notation string
            neighborhood: "moore" (8 neighbors) or "vonneumann" (4 neighbors)
            fade_rate: Decay rate for dead cells (0=instant death, 0.99=long fade)
        """
        super().__init__(size)
        self.rule_str = rule
        self.birth, self.survive = parse_rule(rule)
        self.neighborhood = neighborhood
        self.fade_rate = fade_rate

        # Binary state grid (the actual CA state)
        self.cells = np.zeros((size, size), dtype=np.float64)

    def _count_neighbors(self, grid):
        if self.neighborhood == "vonneumann":
            return _count_neighbors_vonneumann(grid)
        return _count_neighbors_moore(grid)

    def step(self):
        """Advance one generation."""
        neighbors = self._count_neighbors(self.cells)
        neighbors = np.round(neighbors).astype(np.int32)

        new_cells = np.zeros_like(self.cells)
        alive = self.cells > 0.5
        dead = ~alive

        for n in self.birth:
            new_cells[dead & (neighbors == n)] = 1.0
        for n in self.survive:
            new_cells[alive & (neighbors == n)] = 1.0

        self.cells = new_cells

        # Smooth fading: living cells bright, dead cells decay
        alive_mask = self.cells > 0.5
        self.world[alive_mask] = 1.0
        self.world[~alive_mask] *= self.fade_rate

        self.generation += 1
        return self.world

    def apply_feedback(self, feedback):
        """Probabilistic cell birth in feedback regions."""
        # Where feedback is strong enough, randomly birth new cells
        prob = np.clip(feedback * 10.0, 0.0, 1.0)
        birth_mask = (self.cells < 0.5) & (np.random.random(self.cells.shape) < prob)
        self.cells[birth_mask] = 1.0
        self.world[birth_mask] = 1.0

    def set_params(self, rule=None, neighborhood=None, fade_rate=None, **_kw):
        if rule is not None:
            self.rule_str = rule
            self.birth, self.survive = parse_rule(rule)
        if neighborhood is not None:
            self.neighborhood = neighborhood
        if fade_rate is not None:
            self.fade_rate = fade_rate

    def get_params(self):
        return {
            "rule": self.rule_str,
            "neighborhood": self.neighborhood,
            "fade_rate": self.fade_rate,
        }

    def seed(self, seed_type="random", **kwargs):
        density = kwargs.get("density", 0.35)
        if seed_type == "random":
            self._seed_random(density)
        elif seed_type == "center":
            self._seed_center(density)
        elif seed_type == "sparse":
            self._seed_random(0.08)
        else:
            self._seed_random(density)

    def _seed_random(self, density=0.35):
        """Fill grid randomly with given density."""
        self.cells = (np.random.random((self.size, self.size)) < density).astype(np.float64)
        self.world = self.cells.copy()
        self.generation = 0

    def _seed_center(self, density=0.4):
        """Seed a central region."""
        self.cells[:] = 0
        r = self.size // 4
        cy, cx = self.size // 2, self.size // 2
        self.cells[cy-r:cy+r, cx-r:cx+r] = (
            np.random.random((2*r, 2*r)) < density
        ).astype(np.float64)
        self.world = self.cells.copy()
        self.generation = 0

    def add_blob(self, cx, cy, radius=15, value=0.8):
        """Paint alive cells."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        self.cells[mask] = 1.0
        self.world[mask] = 1.0

    def remove_blob(self, cx, cy, radius=15):
        """Erase cells."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = dist < radius
        self.cells[mask] = 0.0
        self.world[mask] = 0.0

    def clear(self):
        self.cells[:] = 0
        self.world[:] = 0
        self.generation = 0

    @property
    def stats(self):
        alive_count = int((self.cells > 0.5).sum())
        total = self.size * self.size
        return {
            "generation": self.generation,
            "mass": float(alive_count),
            "mean": float(self.world.mean()),
            "max": float(self.world.max()),
            "alive_pct": alive_count / total * 100,
        }

    @classmethod
    def get_slider_defs(cls):
        return [
            {"key": "fade_rate", "label": "Fade rate", "section": "RENDERING",
             "min": 0.0, "max": 0.99, "default": 0.92, "fmt": ".2f"},
        ]
