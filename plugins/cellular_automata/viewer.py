"""
Interactive Pygame Viewer for Cellular Automata

Supports multiple CA engines: Lenia, Game of Life, Excitable Media,
and Gray-Scott Reaction-Diffusion. Switch engines and presets via
the side panel.

Features a 4-layer additive color system (Core, Halo, Spark, Memory)
with slowly rotating rainbow hues, optional feedback into the
simulation, and a slow LFO on Lenia growth parameters.

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
from .iridescent import IridescentPipeline
from .presets import (
    PRESETS, PRESET_ORDER, PRESET_ORDERS, ENGINE_ORDER,
    get_preset, get_presets_for_engine,
)
from .controls import ControlPanel, THEME


PANEL_WIDTH = 300
BASE_RES = 512  # Presets are tuned for this resolution

# Engine class registry
ENGINE_CLASSES = {
    "lenia": Lenia,
    "life": Life,
    "excitable": Excitable,
    "gray_scott": GrayScott,
}

ENGINE_LABELS = {
    "lenia": "Lenia",
    "life": "Life",
    "excitable": "Excitable",
    "gray_scott": "Gray-Scott",
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
        # Frequencies chosen to avoid synchronization (prime-ish periods)
        self.mu_lfo = SinusoidalLFO(base_mu, 0.03, frequency_hz=0.015)      # ~67s period
        self.sigma_lfo = SinusoidalLFO(base_sigma, 0.006, frequency_hz=0.012)  # ~83s period
        self.T_lfo = SinusoidalLFO(base_T, base_T * 0.25, frequency_hz=0.014)  # ~71s period

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
        self.mu_lfo.base_value = preset.get("mu", 0.15)
        self.sigma_lfo.base_value = preset.get("sigma", 0.017)
        base_T = preset.get("T", 10)
        self.T_lfo.base_value = base_T
        self.T_lfo.amplitude = base_T * 0.25

    def reset_phase(self):
        """Reset all LFO phases to zero (explicit reseed only)."""
        self.mu_lfo.reset()
        self.sigma_lfo.reset()
        self.T_lfo.reset()


class Viewer:
    def __init__(self, width=900, height=900, sim_size=1024, start_preset="orbium"):
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

        # Iridescent color pipeline
        self.iridescent = IridescentPipeline(sim_size)

        # LFO system for Lenia mu/sigma/T modulation (sinusoidal breathing)
        self.lfo_system = None  # Initialized in _apply_preset

        # Containment field: soft radial decay to keep patterns centered
        self._containment = self._build_containment(sim_size)
        # Center noise mask: gaussian blob for interior perturbation
        self._noise_mask = self._build_noise_mask(sim_size)

        # Engine and preset state
        self.preset_key = start_preset
        preset = get_preset(start_preset)
        self.engine_name = preset["engine"] if preset else "lenia"
        self.engine = None  # Created in _apply_preset

        # Control panel (built after pygame.init in run())
        self.panel = None
        self.sliders = {}
        self.engine_buttons = None
        self.preset_buttons = None

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
        return (0.003 * np.exp(-dist_sq / (2 * 0.15 ** 2))).astype(np.float64)

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
                      if k not in ("engine", "name", "description", "seed", "density")}
            if new_engine_name == "lenia" and "R" in params:
                params["R"] = self._scale_R(params["R"])
            self.engine.set_params(**params)

        # Seed
        seed_kwargs = {}
        if "density" in preset:
            seed_kwargs["density"] = preset["density"]
        self.engine.seed(preset.get("seed", "random"), **seed_kwargs)

        # Reset iridescent pipeline and speed
        self.iridescent.reset(self.sim_size)
        self.speed_accumulator = 0.0

        # Set palette from preset if specified
        if "palette" in preset:
            self.iridescent.set_palette(preset["palette"])

        # Create new LFO system from preset (or None for non-Lenia engines)
        self.lfo_system = LeniaLFOSystem(preset) if new_engine_name == "lenia" else None

        # Rebuild panel if engine changed and panel exists
        if engine_changed and self.panel is not None:
            self._build_panel()
        else:
            self._sync_sliders_from_engine()

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
        if "lfo_speed" in self.sliders and self.lfo_system:
            self.sliders["lfo_speed"].set_value(self.lfo_system.lfo_speed)
        # Iridescent color sliders
        if "tint_r" in self.sliders:
            self.sliders["tint_r"].set_value(self.iridescent.tint_r)
        if "tint_g" in self.sliders:
            self.sliders["tint_g"].set_value(self.iridescent.tint_g)
        if "tint_b" in self.sliders:
            self.sliders["tint_b"].set_value(self.iridescent.tint_b)
        if "brightness" in self.sliders:
            self.sliders["brightness"].set_value(self.iridescent.brightness)

    def _build_panel(self):
        """Build the control panel with engine-specific and common widgets."""
        panel = ControlPanel(self.canvas_w, 0, PANEL_WIDTH, self.canvas_h)
        self.sliders = {}

        # --- Engine selector ---
        panel.add_section("ENGINE")
        engine_labels = [ENGINE_LABELS[e] for e in ENGINE_ORDER]
        engine_idx = ENGINE_ORDER.index(self.engine_name) if self.engine_name in ENGINE_ORDER else 0
        self.engine_buttons = panel.add_button_row(
            engine_labels, selected=engine_idx,
            on_select=self._on_engine_select
        )

        # --- Presets for current engine ---
        panel.add_section("PRESETS")
        preset_keys = get_presets_for_engine(self.engine_name)
        preset_names = [PRESETS[k]["name"] for k in preset_keys]
        preset_idx = preset_keys.index(self.preset_key) if self.preset_key in preset_keys else 0
        self.preset_buttons = panel.add_button_row(
            preset_names, selected=preset_idx,
            on_select=self._on_preset_select
        )

        # --- Engine-specific sliders ---
        current_section = None
        for sdef in self.engine.__class__.get_slider_defs():
            section = sdef.get("section", "PARAMETERS")
            if section != current_section:
                panel.add_section(section)
                current_section = section
            self.sliders[sdef["key"]] = panel.add_slider(
                sdef["label"], sdef["min"], sdef["max"], sdef["default"],
                fmt=sdef.get("fmt", ".3f"),
                step=sdef.get("step"),
                on_change=self._make_param_callback(sdef["key"])
            )

        # --- Iridescent Color Controls ---
        panel.add_section("IRIDESCENT")
        self.sliders["tint_r"] = panel.add_slider(
            "Tint R", 0.0, 2.0, self.iridescent.tint_r, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'tint_r', v)
        )
        self.sliders["tint_g"] = panel.add_slider(
            "Tint G", 0.0, 2.0, self.iridescent.tint_g, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'tint_g', v)
        )
        self.sliders["tint_b"] = panel.add_slider(
            "Tint B", 0.0, 2.0, self.iridescent.tint_b, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'tint_b', v)
        )
        self.sliders["brightness"] = panel.add_slider(
            "Brightness", 0.1, 3.0, self.iridescent.brightness, fmt=".2f",
            on_change=lambda v: setattr(self.iridescent, 'brightness', v)
        )

        # --- LFO Breathing ---
        panel.add_section("LFO BREATHING")
        self.sliders["lfo_speed"] = panel.add_slider(
            "LFO Speed", 0.0, 3.0, 1.0, fmt=".2f",
            on_change=self._on_lfo_speed_change
        )

        # --- Common: Simulation ---
        panel.add_section("SIMULATION")
        self.sliders["speed"] = panel.add_slider(
            "Speed", 0.95, 5.0, self.sim_speed, fmt=".2f",
            on_change=self._on_speed_change
        )
        self.sliders["brush"] = panel.add_slider(
            "Brush size", 3, 80, self.brush_radius, fmt=".0f", step=1,
            on_change=lambda v: setattr(self, 'brush_radius', int(v))
        )

        # --- Actions ---
        panel.add_section("ACTIONS")
        panel.add_button("Reset / Reseed  [R]", on_click=self._on_reset)
        panel.add_spacer(4)
        panel.add_button("Clear World", on_click=self._on_clear)
        panel.add_spacer(4)
        panel.add_button("Screenshot  [S]", on_click=lambda: self._save_screenshot(None))

        self.panel = panel
        self._sync_sliders_from_engine()

    def _make_param_callback(self, key):
        """Create a callback that updates a single engine parameter."""
        def callback(val):
            if self.engine_name == "lenia" and key == "R":
                self.engine.set_params(**{key: int(val)})
            elif key in ("T", "R", "num_states", "threshold"):
                self.engine.set_params(**{key: int(val)})
            else:
                self.engine.set_params(**{key: val})
            # Update LFO base when user adjusts mu/sigma via slider
            if self.lfo_system:
                if key == "mu":
                    self.lfo_system.mu_lfo.base_value = val
                elif key == "sigma":
                    self.lfo_system.sigma_lfo.base_value = val
        return callback

    def _on_engine_select(self, idx, name):
        """Switch to a different engine type."""
        new_engine = ENGINE_ORDER[idx]
        if new_engine == self.engine_name:
            return
        preset_keys = get_presets_for_engine(new_engine)
        if preset_keys:
            self._apply_preset(preset_keys[0])

    def _on_preset_select(self, idx, name):
        preset_keys = get_presets_for_engine(self.engine_name)
        if idx < len(preset_keys):
            self._apply_preset(preset_keys[idx])

    def _on_speed_change(self, val):
        self.sim_speed = val

    def _on_lfo_speed_change(self, val):
        """Update LFO speed multiplier."""
        if self.lfo_system:
            self.lfo_system.lfo_speed = val

    def _on_reset(self):
        preset = get_preset(self.preset_key)
        # Re-apply preset params to ensure clean state
        params = {k: v for k, v in preset.items()
                  if k not in ("engine", "name", "description", "seed", "density")}
        if self.engine_name == "lenia" and "R" in params:
            params["R"] = self._scale_R(params["R"])
        self.engine.set_params(**params)
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
            if self.engine_name == "lenia":
                self.lfo_system.reset_from_preset(preset)
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
        rgb = self.iridescent.render(self.engine.world, dt, lfo_phase=lfo_phase)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
        return surface

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

        rgb = self.iridescent.render(self.engine.world, 0.0)
        save_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
        pygame.image.save(save_surface, path)
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

            # LFO modulation (always runs, even when paused - it's meditative)
            if not self.paused and self.lfo_system:
                self.lfo_system.update(dt)
                modulated = self.lfo_system.get_modulated_params()
                self.engine.set_params(**modulated)

            # Simulation with fractional speed accumulator
            if not self.paused:
                self.speed_accumulator += self.sim_speed
                while self.speed_accumulator >= 1.0:
                    self.engine.step()
                    if self.engine_name == "lenia":
                        # Containment: decay edges to keep pattern centered
                        self.engine.world *= self._containment
                        # Center noise: tiny perturbations keep interior evolving
                        noise = np.random.randn(self.sim_size, self.sim_size) * self._noise_mask
                        self.engine.world = np.clip(self.engine.world + noise, 0.0, 1.0)
                    self.speed_accumulator -= 1.0

                # Auto-reseed if organism dies (never go black)
                if self.engine_name == "lenia":
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

        # Preset selection (1-9) within current engine
        elif pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            preset_keys = get_presets_for_engine(self.engine_name)
            if idx < len(preset_keys):
                self._apply_preset(preset_keys[idx])
                if self.preset_buttons:
                    self.preset_buttons.selected = idx
                    self.preset_buttons._update_active()
