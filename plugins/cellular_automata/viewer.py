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
from .color_layers import ColorLayerSystem, LAYER_DEFS
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

        # Color layer system
        self.layers = ColorLayerSystem(sim_size)

        # LFO state for Lenia mu/sigma/T modulation (state-coupled oscillator)
        self.lfo_phase = 0.0
        self._lfo_base_mu = None
        self._lfo_base_sigma = None
        self._lfo_base_T = None
        self._mu_vel = 0.0       # mu velocity (rate of change)
        self._sigma_vel = 0.0    # sigma velocity
        self._mass_smooth = 0.0  # smoothed organism mass (EMA)

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

        # Reset color layers and speed
        self.layers.reset(self.sim_size)
        self.speed_accumulator = 0.0

        # Reset LFO bases from preset definition (not engine state)
        if new_engine_name == "lenia":
            self._lfo_base_mu = preset.get("mu", 0.15)
            self._lfo_base_sigma = preset.get("sigma", 0.017)
            self._lfo_base_T = preset.get("T", 10)
            self.lfo_phase = 0.0
            self._mu_vel = 0.0
            self._sigma_vel = 0.0
            self._mass_smooth = 0.0
        else:
            self._lfo_base_mu = None
            self._lfo_base_sigma = None
            self._lfo_base_T = None

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

        # --- Color Layers ---
        panel.add_section("COLOR LAYERS")
        initial_colors = self.layers.get_current_colors()
        for i, ldef in enumerate(LAYER_DEFS):
            self.sliders[f"layer_{i}"] = panel.add_color_slider(
                ldef["name"], 0.0, 1.0, self.layers.weights[i],
                swatch_color=initial_colors[i], fmt=".2f",
                on_change=self._make_layer_callback(i)
            )
        self.sliders["feedback"] = panel.add_slider(
            "Feedback", 0.0, 1.0, self.layers.master_feedback, fmt=".2f",
            on_change=lambda v: setattr(self.layers, 'master_feedback', v)
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
            if key == "mu":
                self._lfo_base_mu = val
            elif key == "sigma":
                self._lfo_base_sigma = val
        return callback

    def _make_layer_callback(self, layer_idx):
        """Create a callback for a color layer weight slider."""
        def callback(val):
            self.layers.weights[layer_idx] = val
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
        self.layers.reset()
        self.speed_accumulator = 0.0
        # Reset LFO from preset definition (drift-free)
        if self.engine_name == "lenia":
            self._lfo_base_mu = preset.get("mu", 0.15)
            self._lfo_base_sigma = preset.get("sigma", 0.017)
            self._lfo_base_T = preset.get("T", 10)
            self.lfo_phase = 0.0
            self._mu_vel = 0.0
            self._sigma_vel = 0.0
            self._mass_smooth = 0.0
        self._sync_sliders_from_engine()

    def _on_clear(self):
        self.engine.clear()
        self.layers.reset()
        self.speed_accumulator = 0.0

    def _apply_lfo(self, dt):
        """State-coupled oscillator for organic mu/sigma breathing.

        mu drifts rightward (toward selectivity → contraction),
        sigma drifts leftward (toward narrow tolerance → contraction).
        The organism's mass provides delayed restoring feedback,
        creating a natural predator-prey oscillation cycle.
        """
        if self.engine_name != "lenia" or self._lfo_base_mu is None:
            return
        self.lfo_phase += dt

        # ── Measure organism state (smoothed to prevent jitter) ──
        mass = float(self.engine.world.mean())
        self._mass_smooth += (mass - self._mass_smooth) * min(1.0, 3.0 * dt)

        # Normalized pressure: positive = too big, negative = too small
        target = 0.065
        pressure = (self._mass_smooth - target) / max(target, 0.01)
        pressure = max(-2.0, min(2.0, pressure))

        # Current displacement from base values
        mu_x = self.engine.mu - self._lfo_base_mu
        sigma_x = self.engine.sigma - self._lfo_base_sigma

        # ── mu dynamics (rightward drift) ──
        # Spring pulls back to base; drift pushes right (toward selectivity);
        # mass coupling: too big → push right harder, too small → pull left
        mu_force = (
            0.012                    # rightward drift
            - 0.12 * mu_x           # spring (restoring to base)
            + pressure * 0.8        # mass coupling (strong)
            - 0.20 * self._mu_vel   # damping
        )
        self._mu_vel += mu_force * dt

        # ── sigma dynamics (leftward drift) ──
        # Drift pushes left (narrower); mass pushes back when organism shrinks
        # Coupled to mu displacement: when mu goes right, sigma opens up to compensate
        sigma_force = (
            -0.002                     # leftward drift
            - 0.12 * sigma_x          # spring
            - pressure * 0.12         # mass coupling (strong)
            + mu_x * 0.03             # cross-coupling: mu right → sigma widens
            - 0.20 * self._sigma_vel  # damping
        )
        self._sigma_vel += sigma_force * dt

        # Integrate positions
        new_mu = self.engine.mu + self._mu_vel * dt
        new_sigma = self.engine.sigma + self._sigma_vel * dt

        # Soft walls with bounce (energy loss on reflection)
        mu_lo = self._lfo_base_mu - 0.04
        mu_hi = self._lfo_base_mu + 0.06
        if new_mu < mu_lo:
            new_mu = mu_lo
            self._mu_vel = abs(self._mu_vel) * 0.3
        elif new_mu > mu_hi:
            new_mu = mu_hi
            self._mu_vel = -abs(self._mu_vel) * 0.3

        sigma_lo = max(0.003, self._lfo_base_sigma - 0.008)
        sigma_hi = self._lfo_base_sigma + 0.012
        if new_sigma < sigma_lo:
            new_sigma = sigma_lo
            self._sigma_vel = abs(self._sigma_vel) * 0.3
        elif new_sigma > sigma_hi:
            new_sigma = sigma_hi
            self._sigma_vel = -abs(self._sigma_vel) * 0.3

        self.engine.mu = new_mu
        self.engine.sigma = new_sigma

        # ── T modulation (independent slow sine) ──
        # Oscillates time step: higher T = finer internal structure,
        # lower T = thicker lines/circles. ~45s period.
        if self._lfo_base_T is not None:
            T_offset = math.sin(self.lfo_phase * 0.14) * (self._lfo_base_T * 0.25)
            self.engine.T = max(3, self._lfo_base_T + T_offset)
            self.engine.dt = 1.0 / self.engine.T

    def _update_swatches(self):
        """Update color slider swatches to match current rotating hues."""
        colors = self.layers.get_current_colors()
        for i in range(4):
            key = f"layer_{i}"
            if key in self.sliders:
                self.sliders[key].swatch_color = colors[i]

    def _render_frame(self):
        """Render using the 4-layer color compositing system."""
        signals = self.layers.compute_signals(self.engine.world)
        rgb = self.layers.composite(signals)

        # Apply feedback to simulation if master feedback > 0
        if self.layers.master_feedback > 0.001 and not self.paused:
            feedback = self.layers.compute_feedback(signals)
            self.engine.apply_feedback(feedback)

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

        signals = self.layers.compute_signals(self.engine.world)
        rgb = self.layers.composite(signals)
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
            if not self.paused:
                self._apply_lfo(dt)

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
                        self.layers.reset()
                        self._mu_vel = 0.0
                        self._sigma_vel = 0.0
                        self._mass_smooth = 0.0
                        self.lfo_phase = 0.0

            # Advance hue rotation
            self.layers.advance_time(dt)

            # Render canvas
            screen.fill(THEME["bg"])
            sim_surface = self._render_frame()
            scaled = pygame.transform.smoothscale(sim_surface, (self.canvas_w, self.canvas_h))
            screen.blit(scaled, (0, 0))

            # FPS
            frame_time = time.time() - frame_start
            self.fps_history.append(frame_time)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = 1.0 / max(np.mean(self.fps_history), 0.001)

            self._draw_hud(screen, avg_fps)

            # Panel + update swatches
            if self.panel_visible and self.panel:
                self._update_swatches()
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
