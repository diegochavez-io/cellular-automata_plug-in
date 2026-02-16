"""
Abstract Base Class for Cellular Automaton Engines

All CA engines (Lenia, Game of Life, etc.) implement this interface
so the viewer can work with any engine type interchangeably.
"""

from abc import ABC, abstractmethod
import numpy as np


class CAEngine(ABC):
    """Base class for cellular automaton engines."""

    engine_name = ""   # e.g. "lenia", "life"
    engine_label = ""  # e.g. "Lenia", "Game of Life"

    def __init__(self, size=512):
        self.size = size
        self.world = np.zeros((size, size), dtype=np.float64)
        self.generation = 0

    @abstractmethod
    def step(self):
        """Advance one time step. Returns the world (display) state."""

    def step_n(self, n):
        """Advance n steps. Returns final state."""
        for _ in range(n):
            self.step()
        return self.world

    @abstractmethod
    def set_params(self, **params):
        """Update engine parameters."""

    @abstractmethod
    def get_params(self):
        """Return dict of current parameter values."""

    @abstractmethod
    def seed(self, seed_type="random", **kwargs):
        """Seed the world based on type string."""

    def add_blob(self, cx, cy, radius=15, value=0.8):
        """Paint matter at (cx, cy). Default: smooth falloff."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        influence = np.clip(1.0 - dist / radius, 0, 1) ** 2 * value
        self.world = np.clip(self.world + influence, 0, 1)

    def remove_blob(self, cx, cy, radius=15):
        """Erase matter at (cx, cy). Default: smooth falloff."""
        Y, X = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        influence = np.clip(1.0 - dist / radius, 0, 1) ** 2
        self.world = np.clip(self.world - influence, 0, 1)

    def apply_feedback(self, feedback):
        """Apply color-layer feedback field to the simulation.

        Default implementation adds feedback directly to the world.
        Engines should override for engine-appropriate behavior.

        Args:
            feedback: 2D float array of feedback strength values
        """
        self.world = np.clip(self.world + feedback, 0.0, 1.0)

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

    @classmethod
    @abstractmethod
    def get_slider_defs(cls):
        """Return list of slider definitions for the control panel.

        Each entry is a dict:
            {"key": "mu", "label": "mu (center)", "section": "GROWTH FUNCTION",
             "min": 0.01, "max": 0.40, "default": 0.15,
             "fmt": ".4f", "step": None}
        """
