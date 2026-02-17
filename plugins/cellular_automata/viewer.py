"""
Interactive Pygame Viewer for Cellular Automata

Supports multiple CA engines: Lenia, Game of Life, Excitable Media,
and Gray-Scott Reaction-Diffusion. Switch engines and presets via
the side panel.

Features an iridescent cosine palette pipeline with oil-slick shimmer,
LFO-locked hue sweep, and RGB tint/brightness controls.

Controls:
  SPACE       Pause / Resume
  R           Reset with current preset's seed pattern
  TAB         Toggle control panel
  S           Save screenshot
  H           Toggle HUD overlay
  F           Toggle fullscreen
  Q / ESC     Quit
  Mouse L     Paint matter (on canvas area)
  Mouse R     Erase matter (on canvas area)
"""

import math
import os
import time
import numpy as np
import pygame

from .lenia import Lenia
from .life import Life
from .excitable import Excitable
from .gray_scott import GrayScott
from .cca import CCA
from .smoothlife import SmoothLife
from .mnca import MNCA
from .iridescent import IridescentPipeline
from .presets import (
    PRESETS, PRESET_ORDER, PRESET_ORDERS, ENGINE_ORDER, UNIFIED_ORDER,
    get_preset, get_presets_for_engine,
)
from .controls import ControlPanel, THEME
from .smoothing import SmoothedParameter, LeniaParameterCoupler

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
    from scipy.ndimage import zoom as _scipy_zoom
    from scipy.ndimage import maximum_filter as _scipy_maximum
except ImportError:
    _scipy_gaussian = None
    _scipy_zoom = None
    _scipy_maximum = None


def _dilate_world(world, thickness=5):
    """Thicken structures via max-filter (morphological dilation).

    Unlike blur which smudges detail, dilation makes thin lines physically
    wider while preserving their shape and edges.
    Light blur applied after for softness.
    """
    if _scipy_maximum is None or thickness < 2:
        return world
    # Downsample for speed, dilate, upsample
    factor = 4
    small = world[::factor, ::factor].copy()
    small_thick = max(2, thickness // factor)
    dilated = _scipy_maximum(small, size=small_thick)
    # Light blur for soft edges
    if _scipy_gaussian is not None:
        dilated = _scipy_gaussian(dilated, max(1.0, small_thick * 0.4))
    # Bilinear upsample
    h, w = world.shape
    if _scipy_zoom is not None:
        return _scipy_zoom(dilated, factor, order=1)[:h, :w]
    return np.repeat(np.repeat(dilated, factor, axis=0), factor, axis=1)[:h, :w]


def _blur_world(world, sigma=12):
    """Fast blur via downsample -> gaussian -> bilinear upsample.

    At 4x downsample: 1024->256, blur at sigma/4, bilinear upsample back.
    Bilinear zoom eliminates the blocky grid pattern from np.repeat.
    """
    h, w = world.shape
    factor = 4
    small = world[::factor, ::factor].copy().astype(np.float32)
    small_sigma = max(1.0, sigma / factor)

    if _scipy_gaussian is not None:
        blurred = _scipy_gaussian(small, small_sigma)
    else:
        # Fallback: separable box blur via cumulative sums (3 passes)
        blurred = small
        r = max(1, int(small_sigma * 0.8))
        k = 2 * r + 1
        for _ in range(3):
            p = np.pad(blurred, ((0, 0), (r, r)), mode='reflect')
            cs = np.empty((p.shape[0], p.shape[1] + 1), dtype=np.float32)
            cs[:, 0] = 0
            np.cumsum(p, axis=1, out=cs[:, 1:])
            blurred = (cs[:, k:] - cs[:, :-k]) / k
            p = np.pad(blurred, ((r, r), (0, 0)), mode='reflect')
            cs = np.empty((p.shape[0] + 1, p.shape[1]), dtype=np.float32)
            cs[0, :] = 0
            np.cumsum(p, axis=0, out=cs[1:, :])
            blurred = (cs[k:, :] - cs[:-k, :]) / k

    # Bilinear upsample for smooth result (no blocky grid)
    if _scipy_zoom is not None:
        return _scipy_zoom(blurred, factor, order=1)[:h, :w]
    else:
        return np.repeat(np.repeat(blurred, factor, axis=0), factor, axis=1)[:h, :w]


PANEL_WIDTH = 300
BASE_RES = 512  # Presets are tuned for this resolution

# Engine class registry
ENGINE_CLASSES = {
    "lenia": Lenia,
    "life": Life,
    "excitable": Excitable,
    "gray_scott": GrayScott,
    "cca": CCA,
    "smoothlife": SmoothLife,
    "mnca": MNCA,
}

ENGINE_LABELS = {
    "lenia": "Lenia",
    "life": "Life",
    "excitable": "Excitable",
    "gray_scott": "Gray-Scott",
    "cca": "Cyclic CA",
    "smoothlife": "SmoothLife",
    "mnca": "MNCA",
}


class SinusoidalLFO:
    """Single-parameter phase accumulator with sinusoidal modulation.

    Provides smooth breathing oscillation for one parameter around a base value.
    Phase accumulates continuously based on delta-time for frame-rate independence.
    """

    def __init__(self, base_value, amplitude, frequency_hz=0.01):
        """Initialize LFO.

        Args:
            base_value: Center point of oscillation
            amplitude: Oscillation range (± from base)
            frequency_hz: Oscillation frequency in Hz (default: 0.01 = ~100s period)
        """
        self.base_value = base_value
        self.amplitude = amplitude
        self.frequency_hz = frequency_hz
        self.phase = 0.0

    def update(self, dt):
        """Advance phase by delta-time.

        Args:
            dt: Time elapsed in seconds
        """
        self.phase += 2.0 * math.pi * self.frequency_hz * dt

    def get_value(self):
        """Get current modulated value."""
        return self.base_value + self.amplitude * math.sin(self.phase)

    def reset(self):
        """Reset phase to zero (explicit reseed only)."""
        self.phase = 0.0


class LeniaLFOSystem:
    """Manages three independent sinusoidal LFOs for Lenia parameters.

    Each parameter (mu, sigma, T) oscillates around its base value at different
    frequencies to create organic, non-periodic breathing patterns.
    """

    def __init__(self, preset):
        """Initialize LFO system from preset definition.

        Args:
            preset: Preset dict containing base values for mu, sigma, T
        """
        # Read base values from preset dict (never from engine state)
        base_mu = preset.get("mu", 0.15)
        base_sigma = preset.get("sigma", 0.017)
        base_T = preset.get("T", 10)

        # Create three independent LFOs with different frequencies
        # Amplitudes proportional to base value (15% modulation) for safe breathing
        self.mu_lfo = SinusoidalLFO(base_mu, base_mu * 0.15, frequency_hz=0.015)      # ~67s period
        self.sigma_lfo = SinusoidalLFO(base_sigma, base_sigma * 0.15, frequency_hz=0.012)  # ~83s period
        self.T_lfo = SinusoidalLFO(base_T, base_T * 0.20, frequency_hz=0.014)  # ~71s period

        # Global speed multiplier (adjustable via slider)
        self.lfo_speed = 1.0

    def update(self, dt):
        """Update all LFOs by delta-time scaled by lfo_speed.

        Args:
            dt: Time elapsed in seconds
        """
        scaled_dt = dt * self.lfo_speed
        self.mu_lfo.update(scaled_dt)
        self.sigma_lfo.update(scaled_dt)
        self.T_lfo.update(scaled_dt)

    def get_modulated_params(self):
        """Get current modulated parameter values.

        Returns:
            dict with keys: mu, sigma, T
        """
        T_val = self.T_lfo.get_value()
        return {
            "mu": self.mu_lfo.get_value(),
            "sigma": self.sigma_lfo.get_value(),
            "T": int(max(3, T_val))  # T must be integer, minimum 3
        }

    def reset_from_preset(self, preset):
        """Update base values from preset without resetting phase.

        This allows breathing to continue smoothly when preset parameters change.

        Args:
            preset: Preset dict with potentially updated base values
        """
        base_mu = preset.get("mu", 0.15)
        base_sigma = preset.get("sigma", 0.017)
        base_T = preset.get("T", 10)
        self.mu_lfo.base_value = base_mu
        self.mu_lfo.amplitude = base_mu * 0.15
        self.sigma_lfo.base_value = base_sigma
        self.sigma_lfo.amplitude = base_sigma * 0.15
        self.T_lfo.base_value = base_T
        self.T_lfo.amplitude = base_T * 0.20

    def reset_phase(self):
        """Reset all LFO phases to zero (explicit reseed only)."""
        self.mu_lfo.reset()
        self.sigma_lfo.reset()
        self.T_lfo.reset()


class GrayScottLFOSystem:
    """Single LFO for Gray-Scott feed rate modulation.

    Modulates feed by +/-0.006 around base value with ~30s period.
    Makes holes expand/contract and tendrils wiggle — strong enough
    to push the pattern between regimes for visible structural change.
    """

    def __init__(self, preset):
        base_feed = preset.get("feed", 0.037)
        self.feed_lfo = SinusoidalLFO(base_feed, 0.006, frequency_hz=0.033)  # ~30s period

        # Global speed multiplier (shared with Breath slider)
        self.lfo_speed = 1.0

    def update(self, dt):
        self.feed_lfo.update(dt * self.lfo_speed)

    def get_modulated_params(self):
        return {"feed": self.feed_lfo.get_value()}

    def reset_from_preset(self, preset):
        self.feed_lfo.base_value = preset.get("feed", 0.037)

    def reset_phase(self):
        self.feed_lfo.reset()


class SmoothLifeLFOSystem:
    """LFO system for SmoothLife birth interval modulation.

    Modulates b1 and b2 (birth thresholds) with slow sinusoids.
    Shifting the birth window creates organic breathing — structures
    expand when birth window widens, contract when it narrows.
    """

    def __init__(self, preset):
        base_b1 = preset.get("b1", 0.278)
        base_b2 = preset.get("b2", 0.365)
        self.b1_lfo = SinusoidalLFO(base_b1, 0.015, frequency_hz=1.0 / 70.0)  # ~70s period
        self.b2_lfo = SinusoidalLFO(base_b2, 0.015, frequency_hz=1.0 / 85.0)  # ~85s period
        self.lfo_speed = 1.0

    def update(self, dt):
        scaled_dt = dt * self.lfo_speed
        self.b1_lfo.update(scaled_dt)
        self.b2_lfo.update(scaled_dt)

    def get_modulated_params(self):
        return {
            "b1": self.b1_lfo.get_value(),
            "b2": self.b2_lfo.get_value(),
        }

    def reset_from_preset(self, preset):
        self.b1_lfo.base_value = preset.get("b1", 0.278)
        self.b2_lfo.base_value = preset.get("b2", 0.365)

    def reset_phase(self):
        self.b1_lfo.reset()
        self.b2_lfo.reset()


class MNCALFOSystem:
    """LFO system for MNCA delta modulation.

    Modulates delta (increment magnitude) with a sinusoid.
    Larger delta = more aggressive growth/decay = faster evolution.
    Smaller delta = slower, more subtle changes.
    """

    def __init__(self, preset):
        base_delta = preset.get("delta", 0.05)
        # Modulate delta by +/- 30% of base value
        self.delta_lfo = SinusoidalLFO(base_delta, base_delta * 0.30, frequency_hz=1.0 / 60.0)  # ~60s
        self.lfo_speed = 1.0

    def update(self, dt):
        self.delta_lfo.update(dt * self.lfo_speed)

    def get_modulated_params(self):
        return {"delta": max(0.001, self.delta_lfo.get_value())}

    def reset_from_preset(self, preset):
        base_delta = preset.get("delta", 0.05)
        self.delta_lfo.base_value = base_delta
        self.delta_lfo.amplitude = base_delta * 0.30

    def reset_phase(self):
        self.delta_lfo.reset()


class Viewer:
    def __init__(self, width=900, height=900, sim_size=1024, start_preset="coral"):
        self.canvas_w = width
        self.canvas_h = height
        self.panel_visible = True
        self.sim_size = sim_size
        self.res_scale = sim_size / BASE_RES
        self.running = True
        self.paused = False
        self.show_hud = True
        self.fullscreen = False
        self.brush_radius = 20
        self.fps_history = []

        # Fractional speed system (accumulator pattern)
        self.sim_speed = 1.0  # default: smooth continuous
        self.speed_accumulator = 0.0

        # Render blur (Thickness slider): 0 = sharp, 30 = very soft/thick
        self.render_blur_sigma = 0

        # Iridescent color pipeline
        self.iridescent = IridescentPipeline(sim_size)

        # LFO system for Lenia mu/sigma/T modulation (sinusoidal breathing)
        self.lfo_system = None  # Initialized in _apply_preset
        # LFO system for Gray-Scott feed modulation (breathing)
        self.gs_lfo_system = None  # Initialized in _apply_preset
        # LFO system for SmoothLife birth interval modulation
        self.sl_lfo_system = None  # Initialized in _apply_preset
        # LFO system for MNCA delta modulation
        self.mnca_lfo_system = None  # Initialized in _apply_preset

        # Smoothed parameter infrastructure (EMA drift for organic slider control)
        self.smoothed_params = {}  # key -> SmoothedParameter instances
        self.param_coupler = None  # LeniaParameterCoupler (Lenia only)

        # Containment field: soft radial decay to keep patterns centered
        self._containment = self._build_containment(sim_size)
        # Center noise mask: gaussian blob for interior perturbation
        self._noise_mask = self._build_noise_mask(sim_size)
        # Spatial color offset: radial + angular drives multi-color sweeps
        self._color_offset = self._build_color_offset(sim_size)
        # CCA render mask: soft circle to contain on black background
        self._cca_mask = self._build_cca_mask(sim_size)
        # MNCA containment: aggressive radial mask applied per-step
        self._mnca_containment = self._build_mnca_containment(sim_size)

        # Engine and preset state
        self.preset_key = start_preset
        preset = get_preset(start_preset)
        self.engine_name = preset["engine"] if preset else "lenia"
        self.engine = None  # Created in _apply_preset

        # Control panel (built after pygame.init in run())
        self.panel = None
        self.sliders = {}
        self.preset_buttons = None
        self._hue_value = 0.25  # Hue slider state

        # Create engine and apply preset
        self._apply_preset(self.preset_key)

    @property
    def total_w(self):
        return self.canvas_w + (PANEL_WIDTH if self.panel_visible else 0)

    def _build_containment(self, size):
        """Build a radial decay mask that keeps patterns centered.

        Returns a (size, size) float array: 1.0 at center, gently
        decaying toward edges. Applied each sim step.
        """
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        # Fade starts at 25% from center, aggressive decay kills edges
        fade = np.clip((dist - 0.25) / 0.25, 0.0, 1.0)
        return (1.0 - fade * 0.06).astype(np.float64)

    def _build_noise_mask(self, size):
        """Gaussian mask for center noise injection.

        Strongest at center, fades to zero by ~40% from center.
        Keeps interior structures constantly evolving.
        """
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist_sq = ((X - center) ** 2 + (Y - center) ** 2) / (center * center)
        # Gaussian with sigma ~0.3 of the radius
        return (0.008 * np.exp(-dist_sq / (2 * 0.18 ** 2))).astype(np.float64)

    def _build_color_offset(self, size):
        """Noise-based spatial color offset for organic color pockets.

        Creates natural-looking regions of different colors across the
        organism — like pockets of color in nature, not uniform gradients.
        Uses low-frequency noise blobs + subtle radial variation.
        """
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center

        # Subtle radial: 0.0 at center -> 0.25 at edges
        radial = np.clip(dist, 0, 1) * 0.25

        # Low-frequency noise blobs for color pockets
        # Generate small noise, upsample for large organic blobs
        noise_size = max(8, size // 64)
        noise_small = np.random.randn(noise_size, noise_size).astype(np.float32)
        if _scipy_gaussian is not None:
            noise_small = _scipy_gaussian(noise_small, 1.5)
        if _scipy_zoom is not None:
            noise = _scipy_zoom(noise_small, size / noise_size, order=1)[:size, :size]
        else:
            factor = size // noise_size
            noise = np.repeat(np.repeat(noise_small, factor, axis=0), factor, axis=1)[:size, :size]
        # Normalize to [-0.3, 0.3] range for color pocket variation
        noise_range = max(noise.max() - noise.min(), 0.001)
        noise = (noise - noise.min()) / noise_range * 0.6 - 0.3

        return (radial + noise).astype(np.float32)

    def _build_cca_mask(self, size):
        """Soft circular mask for CCA rendering.

        CCA fills the entire grid by nature. This mask windows the render
        to a circular region so the organism floats on black background.
        Smooth squared falloff creates organic edge.
        """
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        # 1.0 inside 35%, smooth fade to 0 by 55%
        mask = np.clip(1.0 - (dist - 0.35) / 0.20, 0.0, 1.0)
        # Squared falloff for softer edge
        mask = mask ** 2
        return mask.astype(np.float32)

    def _build_mnca_containment(self, size):
        """Radial containment for MNCA.

        Wide gaussian with soft fade — no visible edge artifacts.
        """
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        # Wide gaussian: 1.0 at center, soft fade, ~0.5 at 50% radius
        mask = np.exp(-0.5 * (dist / 0.50) ** 2)
        # Gentle zero toward borders (no hard cliff)
        mask[dist > 0.75] *= np.clip(1.0 - (dist[dist > 0.75] - 0.75) / 0.15, 0.0, 1.0)
        return mask.astype(np.float64)

    def _scale_R(self, base_R):
        """Scale kernel radius for current sim resolution (Lenia only)."""
        return max(5, int(base_R * self.res_scale))

    def _create_engine(self, engine_name, preset):
        """Create a new engine instance from a preset."""
        cls = ENGINE_CLASSES[engine_name]

        if engine_name == "lenia":
            return cls(
                size=self.sim_size,
                R=self._scale_R(preset.get("R", 13)),
                T=preset.get("T", 10),
                mu=preset.get("mu", 0.15),
                sigma=preset.get("sigma", 0.017),
                kernel_peaks=preset.get("kernel_peaks"),
                kernel_widths=preset.get("kernel_widths"),
            )
        elif engine_name == "life":
            return cls(
                size=self.sim_size,
                rule=preset.get("rule", "B3/S23"),
                neighborhood=preset.get("neighborhood", "moore"),
                fade_rate=preset.get("fade_rate", 0.92),
            )
        elif engine_name == "excitable":
            return cls(
                size=self.sim_size,
                num_states=preset.get("num_states", 8),
                threshold=preset.get("threshold", 2),
                neighborhood=preset.get("neighborhood", "moore"),
            )
        elif engine_name == "gray_scott":
            return cls(
                size=self.sim_size,
                feed=preset.get("feed", 0.055),
                kill=preset.get("kill", 0.062),
                Du=preset.get("Du", 0.2097),
                Dv=preset.get("Dv", 0.105),
            )
        elif engine_name == "cca":
            return cls(
                size=self.sim_size,
                range_r=preset.get("range_r", 1),
                threshold=preset.get("threshold", 1),
                num_states=preset.get("num_states", 14),
            )
        elif engine_name == "smoothlife":
            # Scale kernel radii for current sim resolution (presets tuned for BASE_RES)
            ri = max(3, int(preset.get("ri", 8) * self.res_scale))
            ra = max(ri + 3, int(preset.get("ra", 24) * self.res_scale))
            return cls(
                size=self.sim_size,
                ri=ri, ra=ra,
                b1=preset.get("b1", 0.278),
                b2=preset.get("b2", 0.365),
                d1=preset.get("d1", 0.267),
                d2=preset.get("d2", 0.445),
                alpha_n=preset.get("alpha_n", 0.028),
                alpha_m=preset.get("alpha_m", 0.147),
                dt=preset.get("dt", 0.1),
            )
        elif engine_name == "mnca":
            # Scale ring radii for current sim resolution (presets tuned for BASE_RES)
            raw_rings = preset.get("rings") or [(0, 5), (8, 15)]
            scaled_rings = [
                (max(0, int(ir * self.res_scale)), max(2, int(orr * self.res_scale)))
                for ir, orr in raw_rings
            ]
            return cls(
                size=self.sim_size,
                rings=scaled_rings,
                rules=preset.get("rules"),
                delta=preset.get("delta", 0.05),
            )

    def _apply_preset(self, key):
        """Apply a preset, creating a new engine if needed."""
        preset = get_preset(key)
        if preset is None:
            return

        new_engine_name = preset["engine"]
        engine_changed = new_engine_name != self.engine_name

        self.preset_key = key

        self.engine_name = new_engine_name

        if engine_changed or self.engine is None:
            self.engine = self._create_engine(new_engine_name, preset)
        else:
            # Same engine type - just update params
            params = {k: v for k, v in preset.items()
                      if k not in ("engine", "name", "description", "seed", "density", "palette")}
            if new_engine_name == "lenia" and "R" in params:
                params["R"] = self._scale_R(params["R"])
            if new_engine_name == "smoothlife":
                if "ri" in params:
                    params["ri"] = max(3, int(params["ri"] * self.res_scale))
                if "ra" in params:
                    params["ra"] = max(params.get("ri", 6) + 3, int(params["ra"] * self.res_scale))
            if new_engine_name == "mnca" and "rings" in params:
                params["rings"] = [
                    (max(0, int(ir * self.res_scale)), max(2, int(orr * self.res_scale)))
                    for ir, orr in params["rings"]
                ]
            self.engine.set_params(**params)

        # Seed
        seed_kwargs = {}
        if "density" in preset:
            seed_kwargs["density"] = preset["density"]
        self.engine.seed(preset.get("seed", "random"), **seed_kwargs)

        # Reset iridescent pipeline and speed
        self.iridescent.reset(self.sim_size)
        self.speed_accumulator = 0.0

        # Set palette from preset (or reset to default)
        self.iridescent.set_palette(preset.get("palette", "oil_slick"))

        # Create LFO systems from preset (engine-specific)
        self.lfo_system = LeniaLFOSystem(preset) if new_engine_name == "lenia" else None
        self.gs_lfo_system = GrayScottLFOSystem(preset) if new_engine_name == "gray_scott" else None
        self.sl_lfo_system = SmoothLifeLFOSystem(preset) if new_engine_name == "smoothlife" else None
        self.mnca_lfo_system = MNCALFOSystem(preset) if new_engine_name == "mnca" else None

        # Hue cycling speed per engine
        if new_engine_name == "gray_scott":
            self.iridescent._hue_per_breath = 0.20
        elif new_engine_name == "cca":
            self.iridescent._hue_per_breath = 0.12
        elif new_engine_name == "smoothlife":
            self.iridescent._hue_per_breath = 0.10
        elif new_engine_name == "mnca":
            self.iridescent._hue_per_breath = 0.10
        else:
            self.iridescent._hue_per_breath = 0.08

        # Thickness starts at 0 — user controls via slider
        self.render_blur_sigma = 0

        # Create smoothed parameters for all engine sliders
        self.smoothed_params = {}
        for sdef in self.engine.__class__.get_slider_defs():
            param_key = sdef["key"]
            params = self.engine.get_params()
            # Vary time constants for organic independence
            tau = {"mu": 2.0, "sigma": 2.2, "T": 2.5, "R": 2.5}.get(param_key, 2.0)
            sp = SmoothedParameter(params.get(param_key, sdef["default"]), time_constant=tau)
            self.smoothed_params[param_key] = sp

        # Exclude R from smoothing for Lenia — it's structural (rebuilds FFT kernel)
        if new_engine_name == "lenia":
            self.smoothed_params.pop("R", None)

        # Create parameter coupler (Lenia only)
        self.param_coupler = LeniaParameterCoupler(preset) if new_engine_name == "lenia" else None

        # Rebuild panel if engine changed and panel exists (different slider defs)
        if engine_changed and self.panel is not None:
            self._build_panel()
        else:
            self._sync_sliders_from_engine()

        # Update preset button highlight
        if self.preset_buttons and key in UNIFIED_ORDER:
            self.preset_buttons.selected = UNIFIED_ORDER.index(key)
            self.preset_buttons._update_active()


    def _sync_sliders_from_engine(self):
        """Update slider positions to match current engine state."""
        if not self.sliders:
            return
        params = self.engine.get_params()
        for sdef in self.engine.__class__.get_slider_defs():
            key = sdef["key"]
            if key in self.sliders and key in params:
                self.sliders[key].set_value(params[key])
        # Common sliders
        if "speed" in self.sliders:
            self.sliders["speed"].set_value(self.sim_speed)
        if "brush" in self.sliders:
            self.sliders["brush"].set_value(self.brush_radius)
        if "thickness" in self.sliders:
            self.sliders["thickness"].set_value(self.render_blur_sigma)
        # Hue slider
        if "hue" in self.sliders:
            self.sliders["hue"].set_value(getattr(self, '_hue_value', 0.25))
        # Iridescent color sliders (in advanced section)
        if "tint_r" in self.sliders:
            self.sliders["tint_r"].set_value(self.iridescent.tint_r)
        if "tint_g" in self.sliders:
            self.sliders["tint_g"].set_value(self.iridescent.tint_g)
        if "tint_b" in self.sliders:
            self.sliders["tint_b"].set_value(self.iridescent.tint_b)
        if "brightness" in self.sliders:
            self.sliders["brightness"].set_value(self.iridescent.brightness)

    def _build_panel(self):
        """Build the OP-1 style control panel with unified presets."""
        panel = ControlPanel(self.canvas_w, 0, PANEL_WIDTH, self.canvas_h)
        self.sliders = {}

        # --- Unified preset buttons (no engine selector) ---
        panel.add_section("ALGORITHMS")
        preset_names = [PRESETS[k]["name"] for k in UNIFIED_ORDER]
        preset_idx = UNIFIED_ORDER.index(self.preset_key) if self.preset_key in UNIFIED_ORDER else 0
        self.preset_buttons = panel.add_button_row(
            preset_names, selected=preset_idx,
            on_select=self._on_preset_select
        )

        # --- Main controls ---
        panel.add_section("CONTROLS")
        self._hue_value = 0.25
        self.sliders["hue"] = panel.add_slider(
            "Hue", 0.0, 1.0, self._hue_value, fmt=".2f",
            on_change=self._on_hue_change
        )
        self.sliders["brightness"] = panel.add_slider(
            "Brightness", 0.1, 3.0, self.iridescent.brightness, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'brightness', v)
        )
        self.sliders["speed"] = panel.add_slider(
            "Speed", 0.95, 5.0, self.sim_speed, fmt=".2f",
            on_change=self._on_speed_change
        )
        self.sliders["thickness"] = panel.add_slider(
            "Thickness", 0, 30, self.render_blur_sigma, fmt=".0f", step=1,
            on_change=self._on_thickness_change
        )
        self.sliders["brush"] = panel.add_slider(
            "Brush", 3, 80, self.brush_radius, fmt=".0f", step=1,
            on_change=lambda v: setattr(self, 'brush_radius', int(v))
        )

        # --- Engine-specific shape params (in main controls) ---
        engine_sliders = self.engine.__class__.get_slider_defs()
        if engine_sliders:
            panel.add_section("SHAPE")
            for sdef in engine_sliders:
                self.sliders[sdef["key"]] = panel.add_slider(
                    sdef["label"], sdef["min"], sdef["max"], sdef["default"],
                    fmt=sdef.get("fmt", ".3f"),
                    step=sdef.get("step"),
                    on_change=self._make_param_callback(sdef["key"])
                )

        # --- Actions ---
        panel.add_spacer(4)
        panel.add_button("Reseed  [R]", on_click=self._on_reset)
        panel.add_spacer(4)
        panel.add_button("Clear", on_click=self._on_clear)
        panel.add_spacer(4)
        panel.add_button("Screenshot  [S]", on_click=lambda: self._save_screenshot(None))

        # --- Collapsible Advanced section ---
        advanced = panel.add_collapsible_section("Advanced", expanded=False)

        # Tint RGB sliders
        self.sliders["tint_r"] = panel.add_slider_to(
            advanced, "Tint R", 0.0, 2.0, self.iridescent.tint_r, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'tint_r', v)
        )
        self.sliders["tint_g"] = panel.add_slider_to(
            advanced, "Tint G", 0.0, 2.0, self.iridescent.tint_g, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'tint_g', v)
        )
        self.sliders["tint_b"] = panel.add_slider_to(
            advanced, "Tint B", 0.0, 2.0, self.iridescent.tint_b, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'tint_b', v)
        )

        self.panel = panel
        self._sync_sliders_from_engine()

    def _make_param_callback(self, key):
        """Create a callback that updates a single engine parameter.

        For smoothed params (mu, sigma, T): sets EMA target, per-frame code
        feeds smoothed values into LFO bases and engine.
        For R and non-smoothed params: sets engine directly.
        """
        def callback(val):
            if key in self.smoothed_params:
                # Smoothed parameter: set EMA target (per-frame code applies to engine)
                if self.param_coupler and key in ("mu", "sigma"):
                    # Lenia mu/sigma coupling
                    mu_target = self.smoothed_params["mu"].target if "mu" in self.smoothed_params else val
                    sigma_target = self.smoothed_params["sigma"].target if "sigma" in self.smoothed_params else val
                    if key == "mu":
                        mu_target = val
                    else:
                        sigma_target = val
                    coupled_mu, coupled_sigma = self.param_coupler.couple(mu_target, sigma_target)
                    self.smoothed_params["mu"].set_target(coupled_mu)
                    self.smoothed_params["sigma"].set_target(coupled_sigma)
                else:
                    self.smoothed_params[key].set_target(val)
            else:
                # Non-smoothed parameter (e.g. R for Lenia): set engine directly
                if key in ("T", "R", "num_states", "threshold"):
                    self.engine.set_params(**{key: int(val)})
                else:
                    self.engine.set_params(**{key: val})
        return callback

    def _on_preset_select(self, idx, name):
        if idx < len(UNIFIED_ORDER):
            self._apply_preset(UNIFIED_ORDER[idx])

    def _on_speed_change(self, val):
        self.sim_speed = val

    def _on_thickness_change(self, val):
        """Update render blur sigma (Thickness slider)."""
        self.render_blur_sigma = int(val)

    def _on_hue_change(self, val):
        """Update hue via cosine color wheel."""
        self._hue_value = val
        self.iridescent.set_hue_offset(val)

    def _on_reset(self):
        preset = get_preset(self.preset_key)
        # Re-apply preset params to ensure clean state
        params = {k: v for k, v in preset.items()
                  if k not in ("engine", "name", "description", "seed", "density", "palette")}
        if self.engine_name == "lenia" and "R" in params:
            params["R"] = self._scale_R(params["R"])
        if self.engine_name == "smoothlife":
            if "ri" in params:
                params["ri"] = max(3, int(params["ri"] * self.res_scale))
            if "ra" in params:
                params["ra"] = max(params.get("ri", 6) + 3, int(params["ra"] * self.res_scale))
        if self.engine_name == "mnca" and "rings" in params:
            params["rings"] = [
                (max(0, int(ir * self.res_scale)), max(2, int(orr * self.res_scale)))
                for ir, orr in params["rings"]
            ]
        self.engine.set_params(**params)
        # Snap smoothed params to current values (immediate reset, no smoothing)
        for key, sp in self.smoothed_params.items():
            current_params = self.engine.get_params()
            if key in current_params:
                sp.snap(current_params[key])
        # Reseed
        seed_kwargs = {}
        if "density" in preset:
            seed_kwargs["density"] = preset["density"]
        self.engine.seed(preset.get("seed", "random"), **seed_kwargs)
        self.iridescent.reset()
        self.speed_accumulator = 0.0
        # Reset LFO phase (explicit user reseed) and reload base values from preset
        if self.lfo_system:
            self.lfo_system.reset_phase()
            self.lfo_system.reset_from_preset(preset)
        if self.gs_lfo_system:
            self.gs_lfo_system.reset_phase()
            self.gs_lfo_system.reset_from_preset(preset)
        if self.sl_lfo_system:
            self.sl_lfo_system.reset_phase()
            self.sl_lfo_system.reset_from_preset(preset)
        if self.mnca_lfo_system:
            self.mnca_lfo_system.reset_phase()
            self.mnca_lfo_system.reset_from_preset(preset)
        self._sync_sliders_from_engine()

    def _on_clear(self):
        self.engine.clear()
        self.iridescent.reset()
        self.speed_accumulator = 0.0


    def _render_frame(self, dt):
        """Render using iridescent cosine palette pipeline."""
        lfo_phase = None
        if self.lfo_system:
            lfo_phase = self.lfo_system.mu_lfo.phase
        elif self.gs_lfo_system:
            lfo_phase = self.gs_lfo_system.feed_lfo.phase
        elif self.sl_lfo_system:
            lfo_phase = self.sl_lfo_system.b1_lfo.phase
        elif self.mnca_lfo_system:
            lfo_phase = self.mnca_lfo_system.delta_lfo.phase

        if self.engine_name == "gray_scott":
            # Soft rendering: blur world for organic blobby look,
            # density-driven color weights, spatial offset for multi-color sweeps
            world = _blur_world(self.engine.world, sigma=15)
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                color_weights=(0.55, 0.20, 0.25),
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.6)
        elif self.engine_name == "cca":
            # CCA: mask to circular region on black, heavy blur to remove pixelation
            world = self.engine.world * self._cca_mask
            world = _blur_world(world, sigma=28)
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                color_weights=(0.35, 0.25, 0.40),
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.4)
        elif self.engine_name == "smoothlife":
            world = self.engine.world
            if self.render_blur_sigma > 0:
                world = _dilate_world(world, thickness=self.render_blur_sigma)
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.4)
        elif self.engine_name == "mnca":
            world = self.engine.world
            if self.render_blur_sigma > 0:
                world = _dilate_world(world, thickness=self.render_blur_sigma)
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.4)
        else:
            # Lenia + others
            world = self.engine.world
            if self.render_blur_sigma > 0:
                world = _dilate_world(world, thickness=self.render_blur_sigma)
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.4)

        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
        return surface

    def _apply_bloom(self, rgb, sigma=25, intensity=0.5):
        """Colored glow halo via 8x downsample-blur-upsample additive blend.

        128x128x3 blur is fast (~5ms). The heavy downsample is fine for
        bloom since the glow is very diffuse by nature.
        """
        h, w = rgb.shape[:2]
        factor = 8
        small = rgb[::factor, ::factor, :].astype(np.float32)
        small_sigma = max(1.0, sigma / factor)

        if _scipy_gaussian is not None:
            glow = _scipy_gaussian(small, [small_sigma, small_sigma, 0])
        else:
            # Fallback: box blur each channel
            glow = np.empty_like(small)
            r = max(1, int(small_sigma * 0.8))
            k = 2 * r + 1
            for c in range(3):
                ch = small[:, :, c]
                for _ in range(3):
                    p = np.pad(ch, ((0, 0), (r, r)), mode='reflect')
                    cs = np.empty((p.shape[0], p.shape[1] + 1), dtype=np.float32)
                    cs[:, 0] = 0
                    np.cumsum(p, axis=1, out=cs[:, 1:])
                    ch = (cs[:, k:] - cs[:, :-k]) / k
                    p = np.pad(ch, ((r, r), (0, 0)), mode='reflect')
                    cs = np.empty((p.shape[0] + 1, p.shape[1]), dtype=np.float32)
                    cs[0, :] = 0
                    np.cumsum(p, axis=0, out=cs[1:, :])
                    ch = (cs[k:, :] - cs[:-k, :]) / k
                glow[:, :, c] = ch

        glow = np.repeat(np.repeat(glow, factor, axis=0), factor, axis=1)[:h, :w, :]

        result = rgb.astype(np.float32) + glow * intensity
        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)

    def _draw_hud(self, screen, fps):
        if not self.show_hud:
            return

        font = self.hud_font
        stats = self.engine.stats
        preset = get_preset(self.preset_key)
        engine_label = ENGINE_LABELS.get(self.engine_name, self.engine_name)

        line = (f"{engine_label} - {preset['name']}  |  Gen: {stats['generation']:,}  |  "
                f"Alive: {stats['alive_pct']:.1f}%  |  "
                f"{self.sim_size}x{self.sim_size}  |  FPS: {fps:.0f}")
        if self.paused:
            line = "[PAUSED]  " + line

        padding = 6
        bg_height = 24
        bg_surface = pygame.Surface((self.canvas_w, bg_height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 140))
        screen.blit(bg_surface, (0, 0))

        text_surface = font.render(line, True, (210, 215, 225))
        screen.blit(text_surface, (padding + 4, padding))

    def _handle_mouse(self):
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            mx, my = pygame.mouse.get_pos()
            if mx >= self.canvas_w:
                return
            sx = int(mx * self.sim_size / self.canvas_w)
            sy = int(my * self.sim_size / self.canvas_h)
            sim_brush = int(self.brush_radius * self.sim_size / self.canvas_w)
            if buttons[0]:
                self.engine.add_blob(sx, sy, radius=max(sim_brush, 3), value=0.8)
            elif buttons[2]:
                self.engine.remove_blob(sx, sy, radius=max(sim_brush, 3))

    def _save_screenshot(self, screen):
        screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "screenshots"
        )
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(screenshots_dir, f"ca_{self.preset_key}_{timestamp}.png")
        latest_path = os.path.join(screenshots_dir, "latest.png")

        # Render with full pipeline (blur, bloom, color offset) — matches what's on screen
        save_surface = self._render_frame(0.016)
        pygame.image.save(save_surface, path)
        pygame.image.save(save_surface, latest_path)
        print(f"Screenshot saved: {path}")

    def run(self):
        """Main viewer loop."""
        pygame.init()

        screen = pygame.display.set_mode((self.total_w, self.canvas_h), pygame.RESIZABLE)
        pygame.display.set_caption("Cellular Automata")
        clock = pygame.time.Clock()

        self.hud_font = pygame.font.SysFont("menlo", 13)
        self.panel_font = pygame.font.SysFont("menlo", 12)

        self._build_panel()

        last_time = time.time()

        while self.running:
            now = time.time()
            dt = min(now - last_time, 0.1)  # cap dt to avoid jumps
            last_time = now
            frame_start = now

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    continue

                if event.type == pygame.KEYDOWN:
                    self._handle_keydown(event, screen)
                    continue

                if self.panel_visible and self.panel:
                    if self.panel.handle_event(event):
                        continue

            # Mouse painting
            if not (self.panel_visible and pygame.mouse.get_pos()[0] >= self.canvas_w):
                self._handle_mouse()

            if not self.paused:
                # Update smoothed parameters (EMA drift for slider-driven changes)
                if self.smoothed_params:
                    for key, sp in self.smoothed_params.items():
                        sp.update(dt)

                # LFO modulation: smoothed bases → LFO → engine (direct, no dampening)
                if self.lfo_system:
                    # Lenia: Feed smoothed slider values as LFO base values
                    if self.smoothed_params:
                        if "mu" in self.smoothed_params:
                            self.lfo_system.mu_lfo.base_value = self.smoothed_params["mu"].get_value()
                        if "sigma" in self.smoothed_params:
                            self.lfo_system.sigma_lfo.base_value = self.smoothed_params["sigma"].get_value()
                        if "T" in self.smoothed_params:
                            base_T = self.smoothed_params["T"].get_value()
                            self.lfo_system.T_lfo.base_value = base_T
                            self.lfo_system.T_lfo.amplitude = base_T * 0.25
                    self.lfo_system.update(dt)
                    modulated = self.lfo_system.get_modulated_params()
                    self.engine.set_params(**modulated)
                elif self.gs_lfo_system:
                    # Gray-Scott: Feed smoothed feed value as LFO base
                    if "feed" in self.smoothed_params:
                        self.gs_lfo_system.feed_lfo.base_value = self.smoothed_params["feed"].get_value()
                    self.gs_lfo_system.update(dt)
                    modulated = self.gs_lfo_system.get_modulated_params()
                    # Apply non-feed smoothed params directly
                    for k, sp in self.smoothed_params.items():
                        if k != "feed":
                            modulated[k] = sp.get_value()
                    self.engine.set_params(**modulated)
                elif self.sl_lfo_system:
                    # SmoothLife: Feed smoothed b1/b2 as LFO base values
                    if "b1" in self.smoothed_params:
                        self.sl_lfo_system.b1_lfo.base_value = self.smoothed_params["b1"].get_value()
                    if "b2" in self.smoothed_params:
                        self.sl_lfo_system.b2_lfo.base_value = self.smoothed_params["b2"].get_value()
                    self.sl_lfo_system.update(dt)
                    modulated = self.sl_lfo_system.get_modulated_params()
                    # Apply non-LFO smoothed params directly
                    for k, sp in self.smoothed_params.items():
                        if k not in ("b1", "b2"):
                            val = sp.get_value()
                            if k == "ri":
                                val = int(val)
                            modulated[k] = val
                    self.engine.set_params(**modulated)
                elif self.mnca_lfo_system:
                    # MNCA: Feed smoothed delta as LFO base
                    if "delta" in self.smoothed_params:
                        base_delta = self.smoothed_params["delta"].get_value()
                        self.mnca_lfo_system.delta_lfo.base_value = base_delta
                        self.mnca_lfo_system.delta_lfo.amplitude = base_delta * 0.30
                    self.mnca_lfo_system.update(dt)
                    modulated = self.mnca_lfo_system.get_modulated_params()
                    # Apply non-delta smoothed params directly
                    for k, sp in self.smoothed_params.items():
                        if k != "delta":
                            val = sp.get_value()
                            if k in ("inner_r0", "outer_r0"):
                                val = int(val)
                            modulated[k] = val
                    self.engine.set_params(**modulated)
                elif self.smoothed_params:
                    # Other engines: apply smoothed values directly
                    smoothed_vals = {k: sp.get_value() for k, sp in self.smoothed_params.items()}
                    for k in ("T", "R", "num_states", "threshold"):
                        if k in smoothed_vals:
                            smoothed_vals[k] = int(smoothed_vals[k])
                    self.engine.set_params(**smoothed_vals)

            # Simulation with fractional speed accumulator
            if not self.paused:
                self.speed_accumulator += self.sim_speed
                while self.speed_accumulator >= 1.0:
                    self.engine.step()
                    if self.engine_name == "mnca":
                        # MNCA: radial containment (keeps organism from wrapping)
                        self.engine.world *= self._mnca_containment
                        # Center noise: tiny perturbations keep interior evolving
                        noise = np.random.randn(self.sim_size, self.sim_size) * self._noise_mask
                        self.engine.world = np.clip(self.engine.world + noise, 0.0, 1.0)
                        # Periodic blob injection: keeps MNCA constantly reshaping
                        if np.random.random() < 0.02:
                            s = self.sim_size
                            cx = s // 2 + np.random.randint(-s // 6, s // 6)
                            cy = s // 2 + np.random.randint(-s // 6, s // 6)
                            self.engine.add_blob(cx, cy, radius=max(8, s // 30), value=0.4)
                    elif self.engine_name in ("lenia", "smoothlife"):
                        # Containment: gentle decay edges to keep pattern centered
                        self.engine.world *= self._containment
                        # Center noise: perturbations keep interior evolving
                        noise = np.random.randn(self.sim_size, self.sim_size) * self._noise_mask
                        self.engine.world = np.clip(self.engine.world + noise, 0.0, 1.0)
                        # Periodic churn: alternate add/erase to force center reshaping
                        if np.random.random() < 0.03:
                            s = self.sim_size
                            cx = s // 2 + np.random.randint(-s // 5, s // 5)
                            cy = s // 2 + np.random.randint(-s // 5, s // 5)
                            r = max(8, s // 25)
                            if np.random.random() < 0.5:
                                self.engine.add_blob(cx, cy, radius=r, value=0.4)
                            else:
                                self.engine.remove_blob(cx, cy, radius=r)
                    elif self.engine_name == "gray_scott":
                        # Periodic blob injection: seed small V patches near center
                        # to keep the pattern constantly evolving (never fully static)
                        if np.random.random() < 0.03:  # ~3% chance per step
                            s = self.sim_size
                            cx = s // 2 + np.random.randint(-s // 5, s // 5)
                            cy = s // 2 + np.random.randint(-s // 5, s // 5)
                            self.engine.add_blob(cx, cy, radius=s // 25, value=0.3)
                    self.speed_accumulator -= 1.0

                # Auto-reseed if organism dies (never go black)
                if self.engine_name in ("lenia", "smoothlife", "mnca"):
                    mass = float(self.engine.world.mean())
                    if mass < 0.002:
                        preset = get_preset(self.preset_key)
                        seed_kwargs = {}
                        if "density" in preset:
                            seed_kwargs["density"] = preset["density"]
                        self.engine.seed(preset.get("seed", "random"), **seed_kwargs)
                        self.iridescent.reset()
                        if self.lfo_system:
                            self.lfo_system.reset_phase()
                        if self.sl_lfo_system:
                            self.sl_lfo_system.reset_phase()
                        if self.mnca_lfo_system:
                            self.mnca_lfo_system.reset_phase()

            # Render canvas
            screen.fill(THEME["bg"])
            sim_surface = self._render_frame(dt)
            scaled = pygame.transform.smoothscale(sim_surface, (self.canvas_w, self.canvas_h))
            screen.blit(scaled, (0, 0))

            # FPS
            frame_time = time.time() - frame_start
            self.fps_history.append(frame_time)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = 1.0 / max(np.mean(self.fps_history), 0.001)

            self._draw_hud(screen, avg_fps)

            # Panel
            if self.panel_visible and self.panel:
                self.panel.x = self.canvas_w
                self.panel.height = self.canvas_h
                self.panel.draw(screen, self.panel_font)

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def _handle_keydown(self, event, screen):
        key = event.key

        if key in (pygame.K_q, pygame.K_ESCAPE):
            self.running = False

        elif key == pygame.K_SPACE:
            self.paused = not self.paused

        elif key == pygame.K_r:
            self._on_reset()

        elif key == pygame.K_h:
            self.show_hud = not self.show_hud

        elif key == pygame.K_TAB:
            self.panel_visible = not self.panel_visible
            screen = pygame.display.set_mode(
                (self.total_w, self.canvas_h), pygame.RESIZABLE
            )

        elif key == pygame.K_s:
            self._save_screenshot(screen)

        elif key == pygame.K_f:
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                info = pygame.display.Info()
                if self.panel_visible:
                    self.canvas_w = info.current_w - PANEL_WIDTH
                else:
                    self.canvas_w = info.current_w
                self.canvas_h = info.current_h
            else:
                self.canvas_w, self.canvas_h = 900, 900
                screen = pygame.display.set_mode(
                    (self.total_w, self.canvas_h), pygame.RESIZABLE
                )

        # Preset selection (1-9) from unified order
        elif pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            if idx < len(UNIFIED_ORDER):
                self._apply_preset(UNIFIED_ORDER[idx])
                if self.preset_buttons:
                    self.preset_buttons.selected = idx
                    self.preset_buttons._update_active()
