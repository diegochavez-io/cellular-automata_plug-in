"""
Iridescent Color Pipeline for Cellular Automata Visualization

Oil-slick/cuttlefish bioluminescent rendering using cosine palette gradients.
Multi-channel signal mapping (density, edges, velocity) drives spatial color variation.
Global hue sweep locked to LFO breathing phase creates organic color animation.

Replaces the old 4-layer HSV system with unified mathematical color mapping.
"""

import numpy as np


# Cosine palette presets (Inigo Quilez style)
# Each palette has parameters: a, b, c, d (all RGB triplets)
# Formula: color = a + b * cos(2*pi*(c*t + d))

PALETTES = {
    "oil_slick": {
        # Full rainbow oil sheen — many color bands
        "a": np.array([0.50, 0.50, 0.50], dtype=np.float32),
        "b": np.array([0.50, 0.50, 0.50], dtype=np.float32),
        "c": np.array([2.0, 2.0, 2.0], dtype=np.float32),
        "d": np.array([0.00, 0.33, 0.67], dtype=np.float32),
    },
    "cuttlefish": {
        # Vivid organic: cyan, green, yellow, orange, pink, magenta zones
        # High a+b for full gamut, high c for multiple color cycles
        "a": np.array([0.50, 0.50, 0.50], dtype=np.float32),
        "b": np.array([0.50, 0.50, 0.50], dtype=np.float32),
        "c": np.array([1.8, 1.2, 1.5], dtype=np.float32),      # Different rates = rich mixing
        "d": np.array([0.00, 0.25, 0.55], dtype=np.float32),   # Cyan-green-orange-pink shift
    },
    "bioluminescent": {
        # Deep ocean: teal, cyan, peach, coral — aquatic vivid
        "a": np.array([0.45, 0.52, 0.55], dtype=np.float32),   # Teal-biased base
        "b": np.array([0.50, 0.45, 0.42], dtype=np.float32),   # Wide warm range
        "c": np.array([1.5, 1.0, 1.3], dtype=np.float32),
        "d": np.array([0.08, 0.30, 0.55], dtype=np.float32),
    },
    "deep_coral": {
        # Warm vivid: rose, amber, lime, magenta
        "a": np.array([0.52, 0.45, 0.48], dtype=np.float32),
        "b": np.array([0.48, 0.42, 0.45], dtype=np.float32),
        "c": np.array([1.2, 1.8, 1.4], dtype=np.float32),      # Green channel cycles fastest
        "d": np.array([0.00, 0.28, 0.55], dtype=np.float32),
    },
}

# Pre-compute alpha curve LUT (x^0.6 — vivid colors visible at moderate density)
_ALPHA_CURVE = np.power(np.linspace(0, 1, 256, dtype=np.float32), 0.6)


class IridescentPipeline:
    """Cosine palette-based rendering pipeline with multi-channel signal mapping.

    Produces oil-slick/cuttlefish iridescence by mapping simulation state
    (density, edges, velocity) to cosine palette parameters. Global hue sweep
    locked to LFO breathing creates organic color animation.
    """

    def __init__(self, size):
        self.size = size

        # Pre-allocate work buffers
        self.t_buffer = np.zeros((size, size), dtype=np.float32)
        self.display_buffer = np.zeros((size, size, 3), dtype=np.uint8)
        self.edge_buffer = np.zeros((size, size), dtype=np.float32)
        self.vel_buffer = np.zeros((size, size), dtype=np.float32)
        self.prev_world = None
        self._lut_indices = np.zeros((size, size), dtype=np.intp)

        # Combined 2D LUT: (t_idx × alpha_idx) → uint8 RGB
        # Bakes in palette + alpha curve + tint + brightness + uint8 conversion
        # 256×256×3 = 192KB — fits in L2 cache
        self._lut_2d = np.zeros((256 * 256, 3), dtype=np.uint8)
        self._lut_key = None  # Cache key for LUT rebuild

        # Hue state — locked to LFO breathing
        self.hue_phase = 0.0
        self.hue_speed = 0.0003  # Fallback drift for non-LFO engines
        self._prev_lfo_phase = None
        self._lfo_cycle_count = 0
        self._hue_per_breath = 0.08  # 8% of rainbow per breath (~12 breaths for full cycle)

        # User controls (post-render transforms)
        self.tint_r = 1.0
        self.tint_g = 1.0
        self.tint_b = 1.0
        self.brightness = 1.0

        # Smoothed normalization (anti-strobe)
        self._edge_max_smooth = 0.01
        self._vel_max_smooth = 0.01
        self._density_max_smooth = 0.01

        # Current palette — cuttlefish for natural vivid colors
        self.palette_name = "cuttlefish"
        self.palette = PALETTES[self.palette_name]

    def set_palette(self, palette_name):
        if palette_name in PALETTES:
            self.palette_name = palette_name
            self.palette = PALETTES[palette_name]

    def _build_lut_2d(self, a, b, c, d):
        """Build combined (t × alpha) → uint8 LUT with tint/brightness baked in.

        256 palette colors × 256 alpha levels = 65536 entries × 3 bytes = 192KB.
        Eliminates per-pixel float ops: palette eval, alpha, tint, clip, uint8.
        """
        t_vals = np.linspace(0, 1, 256, dtype=np.float32)
        # Palette colors: (256, 3) float32
        # Primary harmonic: broad color sweeps
        colors = a + b * np.cos(2.0 * np.pi * (c * t_vals[:, np.newaxis] + d))
        # Second harmonic: fine-detail color depth (subtle overtone at 2.5x frequency)
        colors += 0.06 * np.cos(2.0 * np.pi * (c * 2.5 * t_vals[:, np.newaxis] + d + 0.3))
        # Third harmonic: micro-detail shimmer (very subtle at 4x frequency)
        colors += 0.03 * np.cos(2.0 * np.pi * (c * 4.0 * t_vals[:, np.newaxis] + d + 0.7))
        np.clip(colors, 0, 1, out=colors)

        tint_bright = np.array(
            [self.tint_r * self.brightness,
             self.tint_g * self.brightness,
             self.tint_b * self.brightness], dtype=np.float32
        )

        alpha_vals = _ALPHA_CURVE  # (256,) float32, 0-1

        # (256 colors, 256 alpha levels, 3 channels) → apply alpha + tint + brightness
        result = (colors[:, np.newaxis, :] *
                  alpha_vals[np.newaxis, :, np.newaxis] *
                  tint_bright * 255.0)
        np.clip(result, 0, 255, out=result)
        self._lut_2d[:] = result.reshape(-1, 3).astype(np.uint8)

    def compute_signals(self, world):
        """Compute edge magnitude and velocity from world state."""
        w = world
        # Horizontal difference into edge_buffer (slicing avoids np.roll copies)
        np.subtract(w[:, :-2], w[:, 2:], out=self.edge_buffer[:, 1:-1])
        self.edge_buffer[:, 0] = w[:, -1] - w[:, 1]
        self.edge_buffer[:, -1] = w[:, -2] - w[:, 0]
        np.square(self.edge_buffer, out=self.edge_buffer)

        # Vertical difference into t_buffer (scratch), then combine
        np.subtract(w[:-2, :], w[2:, :], out=self.t_buffer[1:-1, :])
        self.t_buffer[0, :] = w[-1, :] - w[1, :]
        self.t_buffer[-1, :] = w[-2, :] - w[0, :]
        np.square(self.t_buffer, out=self.t_buffer)

        self.edge_buffer += self.t_buffer
        np.sqrt(self.edge_buffer, out=self.edge_buffer)

        # Smoothed max normalization (anti-strobe)
        edge_max = max(self.edge_buffer.max(), 0.001)
        self._edge_max_smooth = max(edge_max, self._edge_max_smooth * 0.995)
        self.edge_buffer /= self._edge_max_smooth

        # Velocity (temporal change)
        if self.prev_world is not None:
            np.subtract(world, self.prev_world, out=self.vel_buffer)
            np.abs(self.vel_buffer, out=self.vel_buffer)
            vel_max = max(self.vel_buffer.max(), 0.001)
            self._vel_max_smooth = max(vel_max, self._vel_max_smooth * 0.995)
            self.vel_buffer /= self._vel_max_smooth
        else:
            self.vel_buffer[:] = 0.0

        # Store current world for next frame
        if self.prev_world is None:
            self.prev_world = world.copy().astype(np.float32)
        else:
            self.prev_world[:] = world

        return self.edge_buffer, self.vel_buffer

    def compute_color_parameter(self, world, edges, velocity,
                                w_density=0.40, w_velocity=0.30, w_edges=0.30):
        """Map simulation state to gradient parameter t in [0, 1]."""
        density_max = max(world.max(), 0.001)
        self._density_max_smooth = max(density_max, self._density_max_smooth * 0.995)

        # Weighted combination — in-place ops, vel_buffer as scratch
        np.multiply(world, w_density / self._density_max_smooth, out=self.t_buffer)
        self.vel_buffer *= w_velocity  # scale velocity in-place (consumed)
        self.t_buffer += self.vel_buffer
        np.multiply(edges, w_edges, out=self.vel_buffer)  # reuse as scratch
        self.t_buffer += self.vel_buffer
        self.t_buffer %= 1.0
        return self.t_buffer

    def _advance_hue_lfo(self, lfo_phase, dt):
        """Advance hue phase locked to LFO breathing cycles."""
        if lfo_phase is not None:
            # Detect LFO cycle wrap-around
            if self._prev_lfo_phase is not None:
                if lfo_phase < self._prev_lfo_phase - np.pi:
                    self._lfo_cycle_count += 1
            self._prev_lfo_phase = lfo_phase

            # Hue driven by accumulated breath cycles
            lfo_norm = lfo_phase / (2.0 * np.pi)
            self.hue_phase = (
                self._lfo_cycle_count * self._hue_per_breath +
                lfo_norm * self._hue_per_breath * 0.5  # gentle within-breath shimmer
            )
            self.hue_phase %= 1.0
        else:
            # Fallback: very slow constant drift for non-LFO engines
            self.hue_phase += self.hue_speed * dt * 60.0
            self.hue_phase %= 1.0

    def render(self, world, dt, lfo_phase=None, color_weights=None, t_offset=None):
        """Main rendering entry point.

        Args:
            world: Simulation state array (H, W)
            dt: Delta time in seconds
            lfo_phase: Optional LFO phase [0, 2*pi] for breath-locked color
            color_weights: Optional (w_density, w_velocity, w_edges) tuple
            t_offset: Optional (H, W) float32 array added to color parameter
                for spatial color variation (radial/angular sweeps)

        Returns:
            RGB image (H, W, 3) as uint8 array
        """
        # 1. Compute multi-channel signals
        edges, velocity = self.compute_signals(world)

        # 2. Map to color parameter
        if color_weights:
            t = self.compute_color_parameter(world, edges, velocity, *color_weights)
        else:
            t = self.compute_color_parameter(world, edges, velocity)

        # 2b. Spatial color offset for broad multi-color sweeps
        if t_offset is not None:
            self.t_buffer += t_offset
            self.t_buffer %= 1.0

        # 3. Advance hue locked to LFO breathing
        self._advance_hue_lfo(lfo_phase, dt)

        # 4. Build combined 2D LUT (cached — only rebuild when params change)
        d_shifted = self.palette["d"] + self.hue_phase
        lut_key = (round(float(d_shifted[0]), 3), round(float(d_shifted[1]), 3),
                   round(float(d_shifted[2]), 3),
                   round(self.tint_r, 2), round(self.tint_g, 2),
                   round(self.tint_b, 2), round(self.brightness, 2))
        if lut_key != self._lut_key:
            self._build_lut_2d(
                self.palette["a"], self.palette["b"],
                self.palette["c"], d_shifted
            )
            self._lut_key = lut_key

        # 5. Quantize t to [0,255], pre-multiply by 256 for combined index
        np.multiply(t, 255.0, out=self.t_buffer)
        np.clip(self.t_buffer, 0, 255, out=self.t_buffer)
        np.floor(self.t_buffer, out=self.t_buffer)
        np.multiply(self.t_buffer, 256, out=self.t_buffer)

        # 6. Quantize alpha to [0,255] — reuse vel_buffer as scratch
        np.divide(world, max(self._density_max_smooth, 0.001), out=self.vel_buffer)
        np.clip(self.vel_buffer, 0, 1, out=self.vel_buffer)
        np.multiply(self.vel_buffer, 255, out=self.vel_buffer)
        np.floor(self.vel_buffer, out=self.vel_buffer)

        # 7. Combined index: t_idx * 256 + alpha_idx → single LUT lookup
        self.t_buffer += self.vel_buffer
        self._lut_indices[:] = self.t_buffer
        np.take(self._lut_2d, self._lut_indices.ravel(), axis=0,
                out=self.display_buffer.reshape(-1, 3))

        return self.display_buffer

    def set_hue_offset(self, hue):
        """Map single 0-1 hue value to RGB tint via cosine color wheel.

        0.0=warm reds, 0.33=greens, 0.67=cool blues, 1.0=warm again.
        LUT gets rebuilt on next render via property setters.
        """
        import math
        self.tint_r = 1.0 + 0.5 * math.cos(2 * math.pi * hue)
        self.tint_g = 1.0 + 0.5 * math.cos(2 * math.pi * (hue - 0.333))
        self.tint_b = 1.0 + 0.5 * math.cos(2 * math.pi * (hue - 0.667))

    def reset(self, size=None):
        """Reset pipeline state (on engine change or reseed)."""
        if size is not None and size != self.size:
            self.size = size
            self.t_buffer = np.zeros((size, size), dtype=np.float32)
            self.display_buffer = np.zeros((size, size, 3), dtype=np.uint8)
            self.edge_buffer = np.zeros((size, size), dtype=np.float32)
            self.vel_buffer = np.zeros((size, size), dtype=np.float32)
            self._lut_indices = np.zeros((size, size), dtype=np.intp)

        self.prev_world = None
        self._edge_max_smooth = 0.01
        self._vel_max_smooth = 0.01
        self._density_max_smooth = 0.01
        self._prev_lfo_phase = None
        self._lfo_cycle_count = 0
        self._lut_key = None  # Force LUT rebuild
