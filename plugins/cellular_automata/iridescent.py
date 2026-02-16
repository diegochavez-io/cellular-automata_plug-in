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
        "a": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "b": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "c": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "d": np.array([0.00, 0.33, 0.67], dtype=np.float32),
    },
    "cuttlefish": {
        "a": np.array([0.3, 0.5, 0.6], dtype=np.float32),
        "b": np.array([0.4, 0.4, 0.3], dtype=np.float32),
        "c": np.array([1.0, 1.2, 0.8], dtype=np.float32),
        "d": np.array([0.0, 0.25, 0.5], dtype=np.float32),
    },
    "bioluminescent": {
        "a": np.array([0.2, 0.4, 0.5], dtype=np.float32),
        "b": np.array([0.5, 0.5, 0.4], dtype=np.float32),
        "c": np.array([1.5, 1.0, 1.0], dtype=np.float32),
        "d": np.array([0.15, 0.4, 0.6], dtype=np.float32),
    },
    "deep_coral": {
        "a": np.array([0.4, 0.3, 0.5], dtype=np.float32),
        "b": np.array([0.4, 0.5, 0.4], dtype=np.float32),
        "c": np.array([0.8, 1.0, 1.2], dtype=np.float32),
        "d": np.array([0.1, 0.35, 0.55], dtype=np.float32),
    },
}


class IridescentPipeline:
    """Cosine palette-based rendering pipeline with multi-channel signal mapping.

    Produces oil-slick/cuttlefish iridescence by mapping simulation state
    (density, edges, velocity) to cosine palette parameters. Global hue sweep
    locked to LFO breathing creates organic color animation.
    """

    def __init__(self, size):
        self.size = size

        # Pre-allocate all buffers (float32 for performance)
        self.rgb_buffer = np.zeros((size, size, 3), dtype=np.float32)
        self.t_buffer = np.zeros((size, size), dtype=np.float32)
        self.display_buffer = np.zeros((size, size, 3), dtype=np.uint8)
        self.edge_buffer = np.zeros((size, size), dtype=np.float32)
        self.vel_buffer = np.zeros((size, size), dtype=np.float32)
        self.prev_world = None

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

        # Current palette — oil_slick gives widest multi-hue range
        self.palette_name = "oil_slick"
        self.palette = PALETTES[self.palette_name]

    def set_palette(self, palette_name):
        if palette_name in PALETTES:
            self.palette_name = palette_name
            self.palette = PALETTES[palette_name]

    def cosine_palette(self, t, a, b, c, d):
        """Inigo Quilez cosine gradient: color = a + b * cos(2*pi*(c*t + d))"""
        t_expanded = t[:, :, np.newaxis]
        self.rgb_buffer[:] = a + b * np.cos(2.0 * np.pi * (c * t_expanded + d))
        np.clip(self.rgb_buffer, 0.0, 1.0, out=self.rgb_buffer)
        return self.rgb_buffer

    def compute_signals(self, world):
        """Compute edge magnitude and velocity from world state."""
        # Edge magnitude via finite differences
        dx = np.roll(world, 1, axis=1) - np.roll(world, -1, axis=1)
        dy = np.roll(world, 1, axis=0) - np.roll(world, -1, axis=0)
        self.edge_buffer[:] = np.sqrt(dx * dx + dy * dy)

        # Smoothed max normalization (anti-strobe)
        edge_max = max(self.edge_buffer.max(), 0.001)
        self._edge_max_smooth = max(edge_max, self._edge_max_smooth * 0.97)
        self.edge_buffer /= self._edge_max_smooth

        # Velocity (temporal change)
        if self.prev_world is not None:
            self.vel_buffer[:] = np.abs(world - self.prev_world)
            vel_max = max(self.vel_buffer.max(), 0.001)
            self._vel_max_smooth = max(vel_max, self._vel_max_smooth * 0.97)
            self.vel_buffer /= self._vel_max_smooth
        else:
            self.vel_buffer[:] = 0.0

        # Store current world for next frame
        if self.prev_world is None:
            self.prev_world = world.copy().astype(np.float32)
        else:
            self.prev_world[:] = world

        return self.edge_buffer, self.vel_buffer

    def compute_color_parameter(self, world, edges, velocity):
        """Map simulation state to gradient parameter t in [0, 1]."""
        density_max = max(world.max(), 0.001)
        self._density_max_smooth = max(density_max, self._density_max_smooth * 0.97)
        density_norm = world / self._density_max_smooth

        # Edges dominate for rich spatial color variation
        self.t_buffer[:] = (
            0.25 * density_norm +
            0.45 * edges +
            0.30 * velocity
        )
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

    def render(self, world, dt, lfo_phase=None):
        """Main rendering entry point.

        Args:
            world: Simulation state array (H, W)
            dt: Delta time in seconds
            lfo_phase: Optional LFO phase [0, 2*pi] for breath-locked color

        Returns:
            RGB image (H, W, 3) as uint8 array
        """
        # 1. Compute multi-channel signals
        edges, velocity = self.compute_signals(world)

        # 2. Map to color parameter
        t = self.compute_color_parameter(world, edges, velocity)

        # 3. Advance hue locked to LFO breathing
        self._advance_hue_lfo(lfo_phase, dt)

        # 4. Shift palette d parameter by hue phase
        d_shifted = self.palette["d"] + self.hue_phase

        # 5. Generate cosine palette colors
        self.cosine_palette(
            t,
            self.palette["a"],
            self.palette["b"],
            self.palette["c"],
            d_shifted
        )

        # 6. Non-linear alpha for translucent, fluffy organism look
        # Power curve: thin areas become semi-transparent (not binary on/off)
        # This creates the 3D depth / underwater creature feel
        density_norm = world / max(self._density_max_smooth, 0.001)
        alpha = np.power(np.clip(density_norm, 0.0, 1.0), 0.35)
        self.rgb_buffer *= alpha[:, :, np.newaxis]

        # 7. Bioluminescent edge specks — bright dots at high-gradient boundaries
        speck_threshold = 0.65
        speck_mask = (edges > speck_threshold) & (world > 0.02)
        if speck_mask.any():
            speck_intensity = (
                ((edges[speck_mask] - speck_threshold) / (1.0 - speck_threshold)) * 0.35
            )
            self.rgb_buffer[speck_mask] += speck_intensity[:, np.newaxis]

        # 8. Apply RGB tint (post-render multiplication)
        self.rgb_buffer *= np.array(
            [self.tint_r, self.tint_g, self.tint_b], dtype=np.float32
        )

        # 9. Apply brightness (exposure)
        self.rgb_buffer *= self.brightness

        # 10. Convert to uint8 display buffer
        np.clip(self.rgb_buffer, 0.0, 1.0, out=self.rgb_buffer)
        self.display_buffer[:] = (self.rgb_buffer * 255.0).astype(np.uint8)

        return self.display_buffer

    def reset(self, size=None):
        """Reset pipeline state (on engine change or reseed)."""
        if size is not None and size != self.size:
            self.size = size
            self.rgb_buffer = np.zeros((size, size, 3), dtype=np.float32)
            self.t_buffer = np.zeros((size, size), dtype=np.float32)
            self.display_buffer = np.zeros((size, size, 3), dtype=np.uint8)
            self.edge_buffer = np.zeros((size, size), dtype=np.float32)
            self.vel_buffer = np.zeros((size, size), dtype=np.float32)

        self.prev_world = None
        self._edge_max_smooth = 0.01
        self._vel_max_smooth = 0.01
        self._density_max_smooth = 0.01
        self._prev_lfo_phase = None
        self._lfo_cycle_count = 0
