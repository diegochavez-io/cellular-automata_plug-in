"""
CASimulator — Headless simulation core extracted from viewer.py

This module contains all simulation logic (engines, LFOs, flow fields,
containment, coverage management, rendering) with zero pygame dependency.

Designed to be imported by the Scope plugin (Phase 4) and by viewer.py
(which delegates to it while adding the pygame display layer).

Usage:
    from plugins.cellular_automata.simulator import CASimulator
    sim = CASimulator('coral', 512)
    frame = sim.render_float(0.016)  # (H, W, 3) float32 [0, 1]
"""

import math
import time
import numpy as np

from .lenia import Lenia
from .gray_scott import GrayScott
from .smoothlife import SmoothLife
from .mnca import MNCA
from .iridescent import IridescentPipeline
from .presets import (
    PRESETS, PRESET_ORDER, PRESET_ORDERS, ENGINE_ORDER, UNIFIED_ORDER,
    get_preset, get_presets_for_engine,
)
from .smoothing import SmoothedParameter, LeniaParameterCoupler

try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian
    from scipy.ndimage import zoom as _scipy_zoom
    from scipy.ndimage import maximum_filter as _scipy_maximum
    from scipy.ndimage import map_coordinates as _map_coordinates
except ImportError:
    _scipy_gaussian = None
    _scipy_zoom = None
    _scipy_maximum = None
    _map_coordinates = None


# Flow field type names (universal across all engines)
FLOW_KEYS = ["flow_radial", "flow_rotate", "flow_swirl", "flow_bubble",
             "flow_ring", "flow_vortex", "flow_vertical"]

BASE_RES = 512  # Presets are tuned for this resolution

# Engine class registry (only headless-safe engines included)
ENGINE_CLASSES = {
    "lenia": Lenia,
    "gray_scott": GrayScott,
    "smoothlife": SmoothLife,
    "mnca": MNCA,
}


def _dilate_world(world, thickness=5.0):
    """Thicken structures via max-filter + gaussian blur (smooth continuous control).

    Uses gaussian blur for the continuous part of thickness, with max-filter
    kicking in for larger values. This gives smooth slider response at all levels.
    """
    if thickness < 0.5:
        return world
    # Downsample for speed, dilate, upsample
    factor = 4
    small = world[::factor, ::factor].copy()
    sigma = max(0.5, thickness / factor)
    # Max-filter for structural dilation at higher values
    if _scipy_maximum is not None and thickness >= 4:
        size = max(2, int(round(thickness / factor)))
        dilated = _scipy_maximum(small, size=size)
    else:
        dilated = small
    # Gaussian blur for soft, continuous thickness control
    if _scipy_gaussian is not None:
        dilated = _scipy_gaussian(dilated, sigma)
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


# ---------------------------------------------------------------------------
# LFO systems
# ---------------------------------------------------------------------------

class SinusoidalLFO:
    """Single-parameter phase accumulator with sinusoidal modulation.

    Provides smooth breathing oscillation for one parameter around a base value.
    Phase accumulates continuously based on delta-time for frame-rate independence.
    """

    def __init__(self, base_value, amplitude, frequency_hz=0.01):
        """Initialize LFO.

        Args:
            base_value: Center point of oscillation
            amplitude: Oscillation range (+/- from base)
            frequency_hz: Oscillation frequency in Hz (default: 0.01 = ~100s period)
        """
        self.base_value = base_value
        self.amplitude = amplitude
        self.frequency_hz = frequency_hz
        self.phase = 0.0

    def update(self, dt):
        """Advance phase by delta-time."""
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
        """Initialize LFO system from preset definition."""
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
        """Update all LFOs by delta-time scaled by lfo_speed."""
        scaled_dt = dt * self.lfo_speed
        self.mu_lfo.update(scaled_dt)
        self.sigma_lfo.update(scaled_dt)
        self.T_lfo.update(scaled_dt)

    def get_modulated_params(self):
        """Get current modulated parameter values."""
        T_val = self.T_lfo.get_value()
        return {
            "mu": self.mu_lfo.get_value(),
            "sigma": self.sigma_lfo.get_value(),
            "T": int(max(3, T_val))  # T must be integer, minimum 3
        }

    def reset_from_preset(self, preset):
        """Update base values from preset without resetting phase."""
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
    """LFO system for Gray-Scott feed rate modulation.

    Feed LFO: modulates feed by +/-0.006 around base value with ~30s period.
    """

    def __init__(self, preset):
        base_feed = preset.get("feed", 0.037)
        self.feed_lfo = SinusoidalLFO(base_feed, 0.006, frequency_hz=0.033)  # ~30s period
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


# ---------------------------------------------------------------------------
# CASimulator
# ---------------------------------------------------------------------------

class CASimulator:
    """Headless simulation core — zero pygame dependency.

    Encapsulates all CA simulation logic: engine management, LFO breathing,
    flow field advection, containment, coverage management, and frame rendering.

    Designed to be used by:
    - The Scope plugin (imports CASimulator, calls render_float each frame)
    - viewer.py (delegates sim logic to CASimulator, adds pygame display)

    Args:
        preset_key: Initial preset name (e.g. 'coral', 'medusa')
        sim_size: Simulation grid size in pixels (e.g. 512 or 1024)
    """

    def __init__(self, preset_key="coral", sim_size=1024, warmup=True):
        self.sim_size = sim_size
        self.res_scale = sim_size / BASE_RES

        # Fractional speed system (accumulator pattern)
        self.sim_speed = 1.0
        self.speed_accumulator = 0.0

        # Coverage management state
        self._prev_mass = 0.0
        self._stagnant_frames = 0  # Counts consecutive low-change frames
        self._nucleation_counter = 0  # Frame counter for gentle center seeding

        # Velocity-driven perturbation: track previous world for motion detection
        self._prev_world = None  # Pitfall 6: initialize before _apply_preset
        self._perturb_counter = 0  # Run perturbation every Nth step for performance

        # Iridescent color pipeline
        self.iridescent = IridescentPipeline(sim_size)

        # LFO systems (initialized in _apply_preset)
        self.lfo_system = None
        self.gs_lfo_system = None
        self.sl_lfo_system = None
        self.mnca_lfo_system = None

        # Smoothed parameter infrastructure (EMA drift for organic slider control)
        self.smoothed_params = {}
        self.param_coupler = None

        # Containment field: soft radial decay to keep patterns centered
        self._containment = self._build_containment(sim_size)
        # Center noise mask: gaussian blob for interior perturbation
        self._noise_mask = self._build_noise_mask(sim_size)
        # Pre-computed full-res noise pool (no block artifacts, zero per-frame cost)
        self._noise_pool = [np.random.randn(sim_size, sim_size).astype(np.float32) for _ in range(6)]
        self._noise_idx = 0
        # Spatial color offset: radial + angular drives multi-color sweeps
        self._color_offset = self._build_color_offset(sim_size)
        # CCA render mask: soft circle to contain on black background
        self._cca_mask = self._build_cca_mask(sim_size)
        # MNCA containment: aggressive radial mask applied per-step
        self._mnca_containment = self._build_mnca_containment(sim_size)
        # Rotating stir field: slow directional current that prevents center crystallization
        self._stir_dx, self._stir_dy = self._build_stir_field(sim_size)
        self._stir_phase = 0.0

        # Universal flow field system (all engines)
        self._flow_fields = self._build_flow_fields(sim_size)
        # Pre-allocate advection coordinate buffers
        y_coords, x_coords = np.mgrid[:sim_size, :sim_size]
        self._flow_base_y = y_coords.astype(np.float32)
        self._flow_base_x = x_coords.astype(np.float32)
        self._flow_adv_y = np.empty((sim_size, sim_size), dtype=np.float32)
        self._flow_adv_x = np.empty((sim_size, sim_size), dtype=np.float32)
        # Flow param values (shared across all engines)
        for fk in FLOW_KEYS:
            setattr(self, '_' + fk, 0.0)

        # Engine and preset state
        self.preset_key = preset_key
        preset = get_preset(preset_key)
        self.engine_name = preset["engine"] if preset else "lenia"
        self.engine = None

        # Pitfall 2: initialize render_thickness BEFORE _apply_preset
        self.render_thickness = 0.0

        # Hue state (for GS emboss render)
        self._hue_value = 0.25

        # Warmup flag: when False, _apply_preset skips warmup steps.
        # CAPipeline passes warmup=False for fast plugin load.
        self._warmup = warmup

        # Create engine and apply preset
        self._apply_preset(self.preset_key)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def apply_preset(self, key):
        """Public preset switch."""
        self._apply_preset(key)

    def set_runtime_params(self, **kwargs):
        """Set runtime parameters from Scope kwargs.

        Supported keys:
            preset: Switch to named preset
            speed: Set sim_speed
            hue: Set iridescent hue offset (0-1)
            brightness: Set iridescent brightness
            thickness: Set render_thickness
            reseed: Any truthy value triggers engine reseed
            flow_radial, flow_rotate, etc.: Flow field strengths
        """
        for key, val in kwargs.items():
            if key == "preset":
                self.apply_preset(val)
            elif key == "speed":
                self.sim_speed = float(val)
            elif key == "hue":
                self._hue_value = float(val)
                self.iridescent.set_hue_offset(float(val))
            elif key == "brightness":
                self.iridescent.brightness = float(val)
            elif key == "thickness":
                self.render_thickness = float(val)
            elif key == "reseed":
                if val:
                    preset = get_preset(self.preset_key)
                    if preset:
                        seed_kwargs = {}
                        if "density" in preset:
                            seed_kwargs["density"] = preset["density"]
                        self.engine.seed(preset.get("seed", "random"), **seed_kwargs)
            elif key in FLOW_KEYS:
                setattr(self, '_' + key, float(val))

    def step(self, dt) -> np.ndarray:
        """Advance simulation by one rendered frame and return RGB array.

        Runs the full simulation step pipeline:
        - Smoothed parameter updates
        - LFO modulation -> engine parameters
        - Speed accumulator + engine step loop
        - Flow advection, containment, stir/noise/perturbation per step
        - Coverage management
        - Frame rendering

        Args:
            dt: Time elapsed in seconds since last call (e.g. 0.016 for 60fps)

        Returns:
            (H, W, 3) uint8 array in [0, 255]
        """
        # --- Smoothed parameter updates ---
        if self.smoothed_params:
            for key, sp in self.smoothed_params.items():
                sp.update(dt)

        # --- LFO modulation -> engine parameters ---
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
            # Apply non-LFO smoothed params directly
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

        # --- Simulation with fractional speed accumulator ---
        self.speed_accumulator += self.sim_speed
        while self.speed_accumulator >= 1.0:
            self.engine.step()

            # Universal flow advection (after step, before containment)
            # Pitfall 1: GrayScott dual-world — advect BOTH U and V
            if self.engine_name == "gray_scott":
                self.engine.U = self._advect(self.engine.U)
                self.engine.V = self._advect(self.engine.V)
                np.clip(self.engine.U, 0.0, 1.0, out=self.engine.U)
                np.clip(self.engine.V, 0.0, 1.0, out=self.engine.V)
                self.engine.world = self.engine.V
            elif self.engine_name in ("lenia", "smoothlife", "mnca"):
                self.engine.world = self._advect(self.engine.world)

            if self.engine_name == "mnca":
                self.engine.world *= self._mnca_containment
            elif self.engine_name in ("lenia", "smoothlife"):
                self.engine.world *= self._containment
            # Gray-Scott uses its own radial feed mask (don't double-contain)

            # Perturbation + noise + stir: keep organism alive and moving
            # GS excluded — its feed/kill dynamics + LFO + flow handle movement
            if self.engine_name in ("lenia", "smoothlife", "mnca"):
                w = self.engine.world
                has_density = w > 0.03
                self._perturb_counter += 1

                # Velocity perturbation: every 4th step (noise gen is expensive)
                if self._perturb_counter % 4 == 0 and self._prev_world is not None:
                    velocity = np.abs(w - self._prev_world)
                    stagnant = has_density & (velocity < 0.003)
                    if stagnant.any():
                        if self.engine_name in ("lenia", "mnca"):
                            amp = 0.012
                        else:
                            amp = 0.006
                        self.engine.world = np.clip(
                            w + self._fast_noise() * amp * stagnant, 0.0, 1.0)

                # Store snapshot for next step (always, for velocity detection)
                if self._prev_world is None:
                    self._prev_world = w.copy()
                else:
                    self._prev_world[:] = w

                # Center noise: fast low-res noise, masked to density
                noise = self._fast_noise() * self._noise_mask
                noise *= has_density
                self.engine.world = np.clip(
                    self.engine.world + noise, 0.0, 1.0)

                # Rotating stir current (cheap, every step)
                self._stir_phase += 0.005
                stir = (np.cos(self._stir_phase) * self._stir_dx +
                        np.sin(self._stir_phase) * self._stir_dy)
                stir *= has_density
                self.engine.world = np.clip(
                    self.engine.world + stir, 0.0, 1.0)

            self.speed_accumulator -= 1.0

        # --- Coverage management: keep organism alive and within bounds ---
        if self.engine_name in ("lenia", "smoothlife", "mnca", "gray_scott"):
            mass = float(self.engine.world.mean())
            alive_frac = float((self.engine.world > 0.01).sum()) / self.engine.world.size

            # Auto-reseed if organism dies (never go black)
            if mass < 0.002:
                preset = get_preset(self.preset_key)
                seed_kwargs = {}
                if "density" in preset:
                    seed_kwargs["density"] = preset["density"]
                self.engine.seed(preset.get("seed", "random"), **seed_kwargs)
                # Don't reset iridescent — prevents color flash
                if self.lfo_system:
                    self.lfo_system.reset_phase()
                if self.gs_lfo_system:
                    self.gs_lfo_system.reset_phase()
                if self.sl_lfo_system:
                    self.sl_lfo_system.reset_phase()
                if self.mnca_lfo_system:
                    self.mnca_lfo_system.reset_phase()
                self._stagnant_frames = 0

            # GS: skip noise/nucleation — feed/kill dynamics handle coverage
            elif self.engine_name == "gray_scott":
                pass

            # Too dense (> 85%): erode toward center
            elif alive_frac > 0.85:
                if self.engine_name == "mnca":
                    self.engine.world *= self._mnca_containment
                else:
                    self.engine.world *= self._containment
                self.engine.world *= 0.97

            # Stagnation detection (not for GS — uses flow + LFO instead)
            if self.engine_name != "gray_scott":
                delta_mass = abs(mass - self._prev_mass)
                self._prev_mass = mass
                if delta_mass < 0.0001:
                    self._stagnant_frames += 1
                else:
                    self._stagnant_frames = max(0, self._stagnant_frames - 2)

                # After ~2 seconds of stagnation, drop a cluster to stir things up
                if self._stagnant_frames > 120 and 0.10 < alive_frac < 0.85:
                    self._drop_seed_cluster()
                    self._stagnant_frames = 0

            # Nucleation: only when truly dead (< 2%), drop a cluster
            if self.engine_name != "gray_scott" and alive_frac < 0.02:
                self._nucleation_counter += 1
                if self._nucleation_counter >= 30:
                    self._nucleation_counter = 0
                    self._drop_seed_cluster()

        # --- Render frame ---
        return self._render_frame(dt)

    def render_float(self, dt) -> np.ndarray:
        """Advance simulation and return (H, W, 3) float32 [0, 1].

        Calls step(dt) and converts the uint8 output to float32.
        Creates a new array (caller owns the data).

        Args:
            dt: Time elapsed in seconds since last call (e.g. 0.016 for 60fps)

        Returns:
            (H, W, 3) float32 array in [0, 1]
        """
        rgb_uint8 = self.step(dt)
        return rgb_uint8.astype(np.float32) / 255.0

    def run_warmup(self):
        """Run engine-specific warmup steps.

        Called by CAPipeline on first __call__() when constructed with warmup=False.
        Also safe to call after apply_preset() if warmup was skipped.
        """
        if self.engine_name == "lenia":
            for _ in range(200):
                self.engine.step()
        elif self.engine_name == "gray_scott":
            for _ in range(1000):
                self.engine.step()

    # -----------------------------------------------------------------------
    # Field construction
    # -----------------------------------------------------------------------

    def _rebuild_sim_fields(self, new_size):
        """Rebuild all sim-size-dependent masks, fields, and buffers."""
        self.sim_size = new_size
        self.res_scale = new_size / BASE_RES
        self._containment = self._build_containment(new_size)
        self._noise_mask = self._build_noise_mask(new_size)
        self._noise_pool = [np.random.randn(new_size, new_size).astype(np.float32) for _ in range(6)]
        self._noise_idx = 0
        self._color_offset = self._build_color_offset(new_size)
        self._cca_mask = self._build_cca_mask(new_size)
        self._mnca_containment = self._build_mnca_containment(new_size)
        self._stir_dx, self._stir_dy = self._build_stir_field(new_size)
        self._flow_fields = self._build_flow_fields(new_size)
        y_coords, x_coords = np.mgrid[:new_size, :new_size]
        self._flow_base_y = y_coords.astype(np.float32)
        self._flow_base_x = x_coords.astype(np.float32)
        self._flow_adv_y = np.empty((new_size, new_size), dtype=np.float32)
        self._flow_adv_x = np.empty((new_size, new_size), dtype=np.float32)
        self.iridescent.reset(new_size)
        self._prev_world = None

    def _build_containment(self, size):
        """Build a radial decay mask that keeps patterns centered."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        # Gentle fade: starts at 35% from center, gradual over 30%
        fade = np.clip((dist - 0.35) / 0.30, 0.0, 1.0)
        return (1.0 - fade * 0.05).astype(np.float32)

    def _build_noise_mask(self, size):
        """Gaussian mask for center noise injection."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist_sq = ((X - center) ** 2 + (Y - center) ** 2) / (center * center)
        return (0.003 * np.exp(-dist_sq / (2 * 0.18 ** 2))).astype(np.float32)

    def _build_stir_field(self, size):
        """Build directional stir components for rotating current."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dx = (X - center) / center  # -1 to 1
        dy = (Y - center) / center
        dist_sq = dx ** 2 + dy ** 2
        envelope = np.exp(-dist_sq / (2 * 0.25 ** 2))
        amp = 0.010
        stir_dx = (dx * envelope * amp).astype(np.float32)
        stir_dy = (dy * envelope * amp).astype(np.float32)
        return stir_dx, stir_dy

    def _build_color_offset(self, size):
        """Multi-octave noise-based spatial color offset for rich color zones."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center

        # Radial gradient: 0.0 at center -> 0.6 at edges (strong color shift)
        radial = np.clip(dist, 0, 1) * 0.6

        # Layer 1: broad color zones (~64px at 1024)
        noise_size1 = max(8, size // 64)
        n1 = np.random.randn(noise_size1, noise_size1).astype(np.float32)
        if _scipy_gaussian is not None:
            n1 = _scipy_gaussian(n1, 1.5)
        if _scipy_zoom is not None:
            n1 = _scipy_zoom(n1, size / noise_size1, order=1)[:size, :size]
        else:
            f = size // noise_size1
            n1 = np.repeat(np.repeat(n1, f, axis=0), f, axis=1)[:size, :size]
        rng1 = max(n1.max() - n1.min(), 0.001)
        n1 = (n1 - n1.min()) / rng1 * 1.0 - 0.5  # [-0.5, 0.5]

        # Layer 2: finer color zones (~32px at 1024) for detail within organisms
        noise_size2 = max(16, size // 32)
        n2 = np.random.randn(noise_size2, noise_size2).astype(np.float32)
        if _scipy_gaussian is not None:
            n2 = _scipy_gaussian(n2, 1.0)
        if _scipy_zoom is not None:
            n2 = _scipy_zoom(n2, size / noise_size2, order=1)[:size, :size]
        else:
            f = size // noise_size2
            n2 = np.repeat(np.repeat(n2, f, axis=0), f, axis=1)[:size, :size]
        rng2 = max(n2.max() - n2.min(), 0.001)
        n2 = (n2 - n2.min()) / rng2 * 0.6 - 0.3  # [-0.3, 0.3]

        # Combine: radial + broad zones + fine detail
        return (radial + n1 + n2).astype(np.float32)

    def _build_cca_mask(self, size):
        """Soft circular mask for CCA rendering."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        mask = np.clip(1.0 - (dist - 0.35) / 0.20, 0.0, 1.0)
        mask = mask ** 2
        return mask.astype(np.float32)

    def _build_mnca_containment(self, size):
        """Radial containment for MNCA — wide enough to fill 50%+ of frame."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        fade = np.clip((dist - 0.60) / 0.25, 0.0, 1.0)
        return (1.0 - fade * 0.04).astype(np.float32)

    def _build_flow_fields(self, size):
        """Pre-compute 7 velocity fields as (vx, vy) float32 pairs."""
        center = size / 2.0
        Y, X = np.ogrid[:size, :size]
        dx = (X - center) / center  # [-1, 1]
        dy = (Y - center) / center
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dist_safe = np.maximum(dist, 1e-6)

        # Unit radial vector
        rx = dx / dist_safe
        ry = dy / dist_safe

        # Perpendicular (tangent) vector
        tx = -ry
        ty = rx

        fields = {}

        # 1. Radial: push out from center
        fields["flow_radial"] = (rx.astype(np.float32), ry.astype(np.float32))

        # 2. Rotate: perpendicular to radial, weighted by distance
        rot_weight = np.clip(dist, 0, 1)
        fields["flow_rotate"] = (
            (tx * rot_weight).astype(np.float32),
            (ty * rot_weight).astype(np.float32),
        )

        # 3. Swirl: blend rotate (near center) + radial (at edges)
        swirl_blend = np.clip(dist, 0, 1)
        swirl_vx = tx * (1 - swirl_blend * 0.5) + rx * swirl_blend * 0.5
        swirl_vy = ty * (1 - swirl_blend * 0.5) + ry * swirl_blend * 0.5
        swirl_mag = np.sqrt(swirl_vx ** 2 + swirl_vy ** 2)
        swirl_mag_safe = np.maximum(swirl_mag, 1e-6)
        fields["flow_swirl"] = (
            (swirl_vx / swirl_mag_safe * dist).astype(np.float32),
            (swirl_vy / swirl_mag_safe * dist).astype(np.float32),
        )

        # 4. Bubble: radial weighted by gaussian (sigma=0.3)
        bubble_weight = np.exp(-dist ** 2 / (2 * 0.3 ** 2))
        fields["flow_bubble"] = (
            (rx * bubble_weight).astype(np.float32),
            (ry * bubble_weight).astype(np.float32),
        )

        # 5. Ring: radial * sin(dist * 6pi) — concentric push/pull
        ring_mod = np.sin(dist * 6.0 * np.pi)
        fields["flow_ring"] = (
            (rx * ring_mod).astype(np.float32),
            (ry * ring_mod).astype(np.float32),
        )

        # 6. Vortex: perpendicular, dist^2 weighted (accelerating spin)
        vortex_weight = np.clip(dist, 0, 1) ** 2
        fields["flow_vortex"] = (
            (tx * vortex_weight).astype(np.float32),
            (ty * vortex_weight).astype(np.float32),
        )

        # 7. Vertical: uniform downward drift
        fields["flow_vertical"] = (
            np.zeros((size, size), dtype=np.float32),
            np.ones((size, size), dtype=np.float32),
        )

        return fields

    # -----------------------------------------------------------------------
    # Simulation helpers
    # -----------------------------------------------------------------------

    def _advect(self, field):
        """Semi-Lagrangian advection using flow fields."""
        if _map_coordinates is None:
            return field

        total_vx = None
        total_vy = None
        has_flow = False

        for fkey in FLOW_KEYS:
            strength = getattr(self, '_' + fkey)
            if abs(strength) < 0.001:
                continue
            vx, vy = self._flow_fields[fkey]
            if total_vx is None:
                total_vx = vx * strength
                total_vy = vy * strength
            else:
                total_vx = total_vx + vx * strength
                total_vy = total_vy + vy * strength
            has_flow = True

        if not has_flow:
            return field

        # Scale: 0.8px max displacement per step
        total_vx = total_vx * 0.8
        total_vy = total_vy * 0.8

        # Backward trace: source = current_pos - velocity
        np.subtract(self._flow_base_y, total_vy, out=self._flow_adv_y)
        np.subtract(self._flow_base_x, total_vx, out=self._flow_adv_x)

        # Bilinear interpolation with wrapping
        return _map_coordinates(
            field, [self._flow_adv_y, self._flow_adv_x],
            order=1, mode='wrap'
        ).astype(np.float32)

    def _fast_noise(self):
        """Return full-res noise from pre-computed pool. Zero generation cost."""
        self._noise_idx = (self._noise_idx + 1) % len(self._noise_pool)
        return self._noise_pool[self._noise_idx]

    def _drop_center_seed(self):
        """Drop a dense filled blob near the center, sized for the engine."""
        size = self.sim_size
        cx, cy = size // 2, size // 2
        spread = int(size * 0.15)
        sx = cx + np.random.randint(-spread, spread + 1)
        sy = cy + np.random.randint(-spread, spread + 1)

        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - sx)**2 + (Y - sy)**2)

        if self.engine_name == "smoothlife":
            ra = getattr(self.engine, 'ra', 24)
            r = max(15, int(ra * 0.7)) + np.random.randint(-3, 4)
            inner = dist < r
            edge = (dist >= r) & (dist < r + 4)
            fade = (1.0 - (dist[edge] - r) / 4).astype(np.float32)
            values = np.random.random(inner.sum()).astype(np.float32) * 0.2 + 0.75
            self.engine.world[inner] = np.clip(
                self.engine.world[inner] + values, 0.0, 1.0)
            self.engine.world[edge] = np.clip(
                self.engine.world[edge] + fade * 0.5, 0.0, 1.0)
        else:
            base_r = int(12 * self.res_scale)
            r = max(8, base_r + np.random.randint(-2, 3))
            sigma = r / 2.2
            blob = np.exp(-((X - sx)**2 + (Y - sy)**2) / (2 * sigma * sigma))
            mask = dist < r * 2
            self.engine.world[mask] = np.clip(
                self.engine.world[mask] + blob[mask].astype(np.float32) * 0.9,
                0.0, 1.0)

    def _drop_seed_cluster(self):
        """Drop 3-5 seeds clustered near center so they can interact."""
        size = self.sim_size
        cx, cy = size // 2, size // 2
        cluster_offset = int(size * 0.08)
        cluster_cx = cx + np.random.randint(-cluster_offset, cluster_offset + 1)
        cluster_cy = cy + np.random.randint(-cluster_offset, cluster_offset + 1)
        n_seeds = np.random.randint(3, 6)
        for _ in range(n_seeds):
            scatter = int(size * 0.04)
            sx = cluster_cx + np.random.randint(-scatter, scatter + 1)
            sy = cluster_cy + np.random.randint(-scatter, scatter + 1)
            sx = max(0, min(size - 1, sx))
            sy = max(0, min(size - 1, sy))
            self._drop_center_seed_at(sx, sy)

    def _drop_center_seed_at(self, sx, sy):
        """Drop a single dense seed at a specific position."""
        size = self.sim_size
        Y, X = np.ogrid[:size, :size]
        dist = np.sqrt((X - sx)**2 + (Y - sy)**2)

        if self.engine_name == "smoothlife":
            ra = getattr(self.engine, 'ra', 24)
            r = max(15, int(ra * 0.7)) + np.random.randint(-3, 4)
            inner = dist < r
            edge = (dist >= r) & (dist < r + 4)
            fade = (1.0 - (dist[edge] - r) / 4).astype(np.float32)
            values = np.random.random(inner.sum()).astype(np.float32) * 0.2 + 0.75
            self.engine.world[inner] = np.clip(
                self.engine.world[inner] + values, 0.0, 1.0)
            self.engine.world[edge] = np.clip(
                self.engine.world[edge] + fade * 0.5, 0.0, 1.0)
        else:
            base_r = int(12 * self.res_scale)
            r = max(8, base_r + np.random.randint(-2, 3))
            sigma = r / 2.2
            blob = np.exp(-((X - sx)**2 + (Y - sy)**2) / (2 * sigma * sigma))
            mask = dist < r * 2
            self.engine.world[mask] = np.clip(
                self.engine.world[mask] + blob[mask].astype(np.float32) * 0.9,
                0.0, 1.0)

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
        elif engine_name == "gray_scott":
            return cls(
                size=self.sim_size,
                feed=preset.get("feed", 0.055),
                kill=preset.get("kill", 0.062),
                Du=preset.get("Du", 0.2097),
                Dv=preset.get("Dv", 0.105),
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
        else:
            raise ValueError(f"Unknown engine: {engine_name!r}. "
                             f"CASimulator supports: {list(ENGINE_CLASSES.keys())}")

    def _apply_preset(self, key):
        """Apply a preset, creating a new engine if needed."""
        preset = get_preset(key)
        if preset is None:
            return

        new_engine_name = preset["engine"]

        # Only headless-safe engines are supported by CASimulator
        if new_engine_name not in ENGINE_CLASSES:
            raise ValueError(
                f"Preset {key!r} uses engine {new_engine_name!r} which is not "
                f"supported by CASimulator. Supported: {list(ENGINE_CLASSES.keys())}"
            )

        engine_changed = new_engine_name != self.engine_name

        # Sim size is fixed by the constructor (caller controls it).
        # No auto-resize here — the viewer handles that for display optimization.
        # If caller passed sim_size=1024 (default), GS will be slow; use 512 for GS.

        self.preset_key = key
        self.engine_name = new_engine_name

        # Extract flow params from preset
        for fk in FLOW_KEYS:
            setattr(self, '_' + fk, preset.get(fk, 0.0))

        if engine_changed or self.engine is None:
            self.engine = self._create_engine(new_engine_name, preset)
        else:
            # Same engine type - just update params (filter out flow keys)
            params = {k: v for k, v in preset.items()
                      if k not in ("engine", "name", "description", "seed", "density", "palette")
                      and k not in FLOW_KEYS}
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

        # Warmup: pre-run steps so first frames show developed structure.
        if self._warmup:
            if new_engine_name == "lenia":
                for _ in range(200):
                    self.engine.step()
            elif new_engine_name == "gray_scott":
                for _ in range(1000):
                    self.engine.step()

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
        elif new_engine_name == "smoothlife":
            self.iridescent._hue_per_breath = 0.10
        elif new_engine_name == "mnca":
            self.iridescent._hue_per_breath = 0.10
        else:
            self.iridescent._hue_per_breath = 0.08

        # Per-preset controls (hue, brightness, speed, thickness)
        self.render_thickness = preset.get("thickness", 0.0)
        self.sim_speed = preset.get("speed", 1.0)
        if "hue" in preset:
            self._hue_value = preset["hue"]
            self.iridescent.set_hue_offset(preset["hue"])
        if "brightness" in preset:
            self.iridescent.brightness = preset["brightness"]

        # Create smoothed parameters for all engine sliders
        self.smoothed_params = {}
        for sdef in self.engine.__class__.get_slider_defs():
            param_key = sdef["key"]
            params = self.engine.get_params()
            tau = {"mu": 2.0, "sigma": 2.2, "T": 2.5, "R": 2.5}.get(param_key, 2.0)
            sp = SmoothedParameter(params.get(param_key, sdef["default"]), time_constant=tau)
            self.smoothed_params[param_key] = sp

        # Exclude R from smoothing for Lenia — it's structural (rebuilds FFT kernel)
        if new_engine_name == "lenia":
            self.smoothed_params.pop("R", None)

        # Create parameter coupler (Lenia only)
        self.param_coupler = LeniaParameterCoupler(preset) if new_engine_name == "lenia" else None

    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def _render_gs_emboss(self, dt):
        """Render Gray-Scott with heightfield emboss lighting + periodic colormap."""
        V = self.engine.V

        # Contrast stretch: GS V values typically range 0-0.35.
        V_stretch = np.clip(V * 4.0, 0.0, 1.0)

        # Gradients via central differences on raw V (wrap edges)
        dVdx = np.empty_like(V)
        dVdy = np.empty_like(V)
        dVdx[:, 1:-1] = (V[:, 2:] - V[:, :-2]) * 0.5
        dVdx[:, 0] = (V[:, 1] - V[:, -1]) * 0.5
        dVdx[:, -1] = (V[:, 0] - V[:, -2]) * 0.5
        dVdy[1:-1, :] = (V[2:, :] - V[:-2, :]) * 0.5
        dVdy[0, :] = (V[1, :] - V[-1, :]) * 0.5
        dVdy[-1, :] = (V[0, :] - V[-2, :]) * 0.5

        # Heightfield normals: N = normalize(dVdx * h, dVdy * h, -1)
        height_scale = np.float32(12.0)
        nx = dVdx * height_scale
        ny = dVdy * height_scale
        nz_val = np.float32(-1.0)
        nmag = np.sqrt(nx * nx + ny * ny + nz_val * nz_val)
        nmag = np.maximum(nmag, np.float32(1e-6))
        inv_nmag = np.float32(1.0) / nmag
        nx *= inv_nmag
        ny *= inv_nmag
        nz = nz_val * inv_nmag

        # Light from upper-left
        lx, ly, lz = -0.5, -0.5, 1.0
        lmag = math.sqrt(lx * lx + ly * ly + lz * lz)
        lx /= lmag; ly /= lmag; lz /= lmag

        # Diffuse lighting
        diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

        # Specular (Blinn-Phong, view direction = (0, 0, -1))
        hx, hy, hz = lx, ly, lz - 1.0
        hmag = math.sqrt(hx * hx + hy * hy + hz * hz)
        hx /= hmag; hy /= hmag; hz /= hmag
        spec = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0)
        np.power(spec, 32, out=spec)

        # Periodic colormap on contrast-stretched V
        frequency = np.float32(2.5)
        phase = np.float32(self._hue_value * 2.0 * math.pi)
        t = V_stretch * frequency

        # Cosine palette: warm organic tones with hue rotation
        two_pi = np.float32(2.0 * math.pi)
        rgb = np.empty((*V.shape, 3), dtype=np.float32)
        d_offsets = (0.00, 0.33, 0.67)
        for i in range(3):
            rgb[:, :, i] = 0.5 + 0.5 * np.cos(two_pi * t + phase + d_offsets[i] * two_pi)

        # Combine: (ambient + diffuse) * color + specular
        lighting = (np.float32(0.15) + diffuse * np.float32(0.85))
        result = lighting[:, :, np.newaxis] * rgb

        # Warm specular highlights
        spec_contrib = spec * np.float32(0.35)
        result[:, :, 0] += spec_contrib
        result[:, :, 1] += spec_contrib * 0.92
        result[:, :, 2] += spec_contrib * 0.85

        # Black background where V is negligible
        alive_mask = (V > 0.005).astype(np.float32)
        result *= alive_mask[:, :, np.newaxis]

        # Apply brightness control
        result *= self.iridescent.brightness

        np.clip(result, 0.0, 1.0, out=result)
        result *= 255.0
        return result.astype(np.uint8)

    def _render_frame(self, dt) -> np.ndarray:
        """Render current simulation state to (H, W, 3) uint8 array.

        Returns RGB data (no pygame Surface).
        """
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
            # GS V values typically range 0-0.35. Dead zone + contrast stretch.
            world = np.clip((self.engine.world - 0.015) * 3.5, 0.0, 1.0)
        elif self.engine_name == "smoothlife":
            # SmoothLife: softer contrast curve
            world = np.power(self.engine.world, 1.4)
            if _scipy_gaussian is not None:
                world = _scipy_gaussian(world, 2.0)
        else:
            # MNCA, Lenia — direct render
            world = self.engine.world

        # Thickness: dilate structures (0 = raw)
        if self.render_thickness > 0.5:
            world = _dilate_world(world, self.render_thickness)

        # Render through iridescent pipeline
        if self.engine_name == "gray_scott":
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                t_offset=self._color_offset,
            )
        elif self.engine_name == "smoothlife":
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                color_weights=(0.30, 0.25, 0.45),
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.25)
        else:
            # Lenia/MNCA: edge-heavy color weights for 3D depth
            rgb = self.iridescent.render(
                world, dt, lfo_phase=lfo_phase,
                color_weights=(0.30, 0.25, 0.45),
                t_offset=self._color_offset,
            )
            rgb = self._apply_bloom(rgb, intensity=0.20)

        # rgb is a (H, W, 3) uint8 from iridescent.render (display_buffer)
        # Return a copy so caller owns the data (display_buffer is reused)
        return rgb.copy()

    def _apply_bloom(self, rgb, sigma=25, intensity=0.5):
        """Colored glow halo via 8x downsample-blur-upsample additive blend."""
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

        # Scale intensity at small res, convert to uint8 before upsample
        np.multiply(glow, intensity, out=glow)
        np.clip(glow, 0, 255, out=glow)
        glow_u8 = glow.astype(np.uint8)

        # Upsample at uint8 (3x less memory than float32 repeat)
        glow_up = np.repeat(np.repeat(glow_u8, factor, axis=0), factor, axis=1)[:h, :w, :]

        # Saturating add via uint16
        return np.minimum(
            rgb.astype(np.uint16) + glow_up, 255
        ).astype(np.uint8)
