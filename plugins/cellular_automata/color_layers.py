"""
Multi-Layer Color System for Cellular Automata Visualization

Four independent color layers, each revealing a different dimension
of the simulation:
  - Core:   Cell density (world values)
  - Halo:   Edge magnitude (gradient of world)
  - Spark:  Temporal change (velocity)
  - Memory: Trail buffer (exponential moving average)

Layers are additively composited with slowly rotating rainbow hues
and can feed back into the simulation.
"""

import math
import numpy as np


# Layer definitions: name, default weight, initial hue (degrees)
LAYER_DEFS = [
    {"name": "Core",   "default_weight": 0.80, "hue": 330},
    {"name": "Halo",   "default_weight": 0.60, "hue":  60},
    {"name": "Spark",  "default_weight": 0.40, "hue": 170},
    {"name": "Memory", "default_weight": 0.70, "hue": 270},
]

# Feedback coefficients per layer
_FEEDBACK_COEFFS = np.array([0.003, 0.002, 0.002, 0.001], dtype=np.float32)

DEFAULT_MASTER_FEEDBACK = 0.30


def _hsv_to_rgb(h, s, v):
    """Convert HSV to RGB tuple. h in degrees, s/v in [0,1]."""
    h = h % 360
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0.0
    elif h < 120:
        r, g, b = x, c, 0.0
    elif h < 180:
        r, g, b = 0.0, c, x
    elif h < 240:
        r, g, b = 0.0, x, c
    elif h < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return ((r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0)


class ColorLayerSystem:
    """Manages the 4 color layers, compositing, and feedback computation."""

    def __init__(self, size):
        self.size = size
        self.weights = [d["default_weight"] for d in LAYER_DEFS]
        self.master_feedback = DEFAULT_MASTER_FEEDBACK

        # Hue rotation state
        self.hue_time = 0.0
        self.hue_speed = 2.5  # degrees per second (~144s per full rotation)
        self._base_hues = [d["hue"] for d in LAYER_DEFS]
        self._sat = 0.50
        self._val = 1.0
        self._color_matrix = np.zeros((4, 3), dtype=np.float32)
        self._update_colors()

        # Buffers (float32 for speed)
        self.prev_world = None
        self.memory = np.zeros((size, size), dtype=np.float32)
        self.memory_decay = 0.97

        # Smoothed normalization (prevents strobing)
        self._edge_max_smooth = 0.01
        self._vel_max_smooth = 0.01

        # Pre-allocated work buffers
        self._rgb = np.zeros((size, size, 3), dtype=np.float32)
        self._signals = np.zeros((4, size, size), dtype=np.float32)

    def _update_colors(self):
        """Recompute color matrix from current hue rotation."""
        for i in range(4):
            hue = (self._base_hues[i] + self.hue_time) % 360
            self._color_matrix[i] = _hsv_to_rgb(hue, self._sat, self._val)

    def advance_time(self, dt):
        """Advance hue rotation by dt seconds."""
        self.hue_time += self.hue_speed * dt
        self._update_colors()

    def get_current_colors(self):
        """Return current layer colors as list of RGB tuples (for UI swatches)."""
        return [tuple(int(c) for c in self._color_matrix[i]) for i in range(4)]

    def reset(self, size=None):
        """Reset all buffers (on engine change / reseed)."""
        if size is not None and size != self.size:
            self.size = size
            self.memory = np.zeros((size, size), dtype=np.float32)
            self._rgb = np.zeros((size, size, 3), dtype=np.float32)
            self._signals = np.zeros((4, size, size), dtype=np.float32)
        else:
            self.memory[:] = 0
        self.prev_world = None
        self._edge_max_smooth = 0.01
        self._vel_max_smooth = 0.01

    def compute_signals(self, world):
        """Compute the 4 signal fields from the current world state.

        Writes into self._signals[0..3] and returns the array.
        """
        S = self._signals
        w = world

        # Core: raw density
        np.clip(w, 0.0, 1.0, out=S[0])

        # Halo: edge magnitude via finite differences
        gx = np.roll(w, -1, axis=1)
        gx -= np.roll(w, 1, axis=1)
        gy = np.roll(w, -1, axis=0)
        gy -= np.roll(w, 1, axis=0)
        np.multiply(gx, gx, out=S[1])
        S[1] += gy * gy
        np.sqrt(S[1], out=S[1])
        # Smoothed max: jumps up instantly, decays slowly (prevents strobing)
        edge_max = float(S[1].max())
        self._edge_max_smooth = max(edge_max, self._edge_max_smooth * 0.97)
        norm = max(self._edge_max_smooth, 0.01)
        S[1] *= (1.0 / norm)
        np.clip(S[1], 0.0, 1.0, out=S[1])

        # Spark: temporal change
        if self.prev_world is not None:
            np.subtract(w, self.prev_world, out=S[2])
            np.abs(S[2], out=S[2])
            # Smoothed max for velocity too
            vel_max = float(S[2].max())
            self._vel_max_smooth = max(vel_max, self._vel_max_smooth * 0.97)
            norm_v = max(self._vel_max_smooth, 0.005)
            S[2] *= (1.0 / norm_v)
            np.clip(S[2], 0.0, 1.0, out=S[2])
        else:
            S[2][:] = 0

        # Store current world for next frame's velocity
        if self.prev_world is None:
            self.prev_world = w.astype(np.float32).copy()
        else:
            np.copyto(self.prev_world, w)

        # Memory: EMA
        self.memory *= self.memory_decay
        self.memory += (1.0 - self.memory_decay) * w
        np.clip(self.memory, 0.0, 1.0, out=S[3])

        return S

    def composite(self, signals):
        """Additively blend the 4 layers into an RGB image.

        Args:
            signals: (4, H, W) float32 array from compute_signals

        Returns:
            (H, W, 3) uint8 RGB image
        """
        rgb = self._rgb
        rgb[:] = 0

        w = np.array(self.weights, dtype=np.float32)
        wc = self._color_matrix * w[:, np.newaxis]  # (4, 3)

        for i in range(4):
            if w[i] < 0.001:
                continue
            s = signals[i]
            rgb[:, :, 0] += s * wc[i, 0]
            rgb[:, :, 1] += s * wc[i, 1]
            rgb[:, :, 2] += s * wc[i, 2]

        np.clip(rgb, 0, 255, out=rgb)
        return rgb.astype(np.uint8)

    def compute_feedback(self, signals):
        """Compute the feedback field to apply to the simulation."""
        w = np.array(self.weights, dtype=np.float32)
        coeffs = w * _FEEDBACK_COEFFS * self.master_feedback
        return np.tensordot(coeffs, signals, axes=([0], [0]))
