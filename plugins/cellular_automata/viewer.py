"""
Interactive Pygame Viewer for Cellular Automata

Delegates all simulation to CASimulator. This module is a thin pygame
display wrapper: it owns the window, event loop, control panel, and HUD.

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

import os
import time
import numpy as np
import pygame

from .simulator import (
    CASimulator, FLOW_KEYS, ENGINE_CLASSES,
)

ENGINE_LABELS = {
    "lenia": "Lenia",
    "life": "Life",
    "excitable": "Excitable",
    "gray_scott": "Gray-Scott",
    "cca": "Cyclic CA",
    "smoothlife": "SmoothLife",
    "mnca": "MNCA",
}
from .presets import (
    PRESETS, PRESET_ORDER, PRESET_ORDERS, ENGINE_ORDER, UNIFIED_ORDER,
    get_preset, get_presets_for_engine,
)
from .controls import ControlPanel, THEME

# Slider definitions for flow fields (display-only, used to build panel)
FLOW_SLIDER_DEFS = [
    {"key": "flow_radial",   "label": "Radial"},
    {"key": "flow_rotate",   "label": "Rotate"},
    {"key": "flow_swirl",    "label": "Swirl"},
    {"key": "flow_bubble",   "label": "Bubble"},
    {"key": "flow_ring",     "label": "Ring"},
    {"key": "flow_vortex",   "label": "Vortex"},
    {"key": "flow_vertical", "label": "Vertical"},
]

PANEL_WIDTH = 300


class Viewer:
    def __init__(self, width=900, height=900, sim_size=1024, start_preset="coral"):
        self.canvas_w = width
        self.canvas_h = height
        self.panel_visible = True
        self.running = True
        self.paused = False
        self.show_hud = True
        self.fullscreen = False
        self.brush_radius = 20
        self.fps_history = []

        # Delegate ALL simulation to CASimulator
        self.simulator = CASimulator(preset_key=start_preset, sim_size=sim_size)

        # Control panel (built after pygame.init in run())
        self.panel = None
        self.sliders = {}
        self.preset_buttons = None
        self._hue_value = 0.25  # Hue slider state (display-side tracking)

    @property
    def total_w(self):
        return self.canvas_w + (PANEL_WIDTH if self.panel_visible else 0)

    def _sync_sliders_from_engine(self):
        """Update slider positions to match current engine/simulator state."""
        if not self.sliders:
            return
        params = self.simulator.engine.get_params()
        for sdef in self.simulator.engine.__class__.get_slider_defs():
            key = sdef["key"]
            if key in self.sliders and key in params:
                self.sliders[key].set_value(params[key])
        # Common sliders
        if "speed" in self.sliders:
            self.sliders["speed"].set_value(self.simulator.sim_speed)
        if "thickness" in self.sliders:
            self.sliders["thickness"].set_value(self.simulator.render_thickness)
        if "brush" in self.sliders:
            self.sliders["brush"].set_value(self.brush_radius)
        # Hue slider
        if "hue" in self.sliders:
            self.sliders["hue"].set_value(self._hue_value)
        # Iridescent color sliders
        if "tint_r" in self.sliders:
            self.sliders["tint_r"].set_value(self.simulator.iridescent.tint_r)
        if "tint_g" in self.sliders:
            self.sliders["tint_g"].set_value(self.simulator.iridescent.tint_g)
        if "tint_b" in self.sliders:
            self.sliders["tint_b"].set_value(self.simulator.iridescent.tint_b)
        if "brightness" in self.sliders:
            self.sliders["brightness"].set_value(self.simulator.iridescent.brightness)
        # Flow sliders (read from simulator)
        for fk in FLOW_KEYS:
            if fk in self.sliders:
                self.sliders[fk].set_value(getattr(self.simulator, '_' + fk))

    def _build_panel(self):
        """Build the OP-1 style control panel with unified presets."""
        panel = ControlPanel(self.canvas_w, 0, PANEL_WIDTH, self.canvas_h)
        self.sliders = {}

        # --- Unified preset buttons (no engine selector) ---
        panel.add_section("ALGORITHMS")
        preset_names = [PRESETS[k]["name"] for k in UNIFIED_ORDER]
        preset_idx = UNIFIED_ORDER.index(self.simulator.preset_key) if self.simulator.preset_key in UNIFIED_ORDER else 0
        self.preset_buttons = panel.add_button_row(
            preset_names, selected=preset_idx,
            on_select=self._on_preset_select
        )

        # --- Main controls ---
        panel.add_section("CONTROLS")
        self._hue_value = self.simulator._hue_value
        self.sliders["hue"] = panel.add_slider(
            "Hue", 0.0, 1.0, self._hue_value, fmt=".2f",
            on_change=self._on_hue_change
        )
        self.sliders["brightness"] = panel.add_slider(
            "Brightness", 0.1, 3.0, self.simulator.iridescent.brightness, fmt=".2f",
            on_change=lambda v: setattr(self.simulator.iridescent, 'brightness', v)
        )
        self.sliders["speed"] = panel.add_slider(
            "Speed", 0.95, 5.0, self.simulator.sim_speed, fmt=".2f",
            on_change=self._on_speed_change
        )
        self.sliders["thickness"] = panel.add_slider(
            "Thickness", 0.0, 20.0, self.simulator.render_thickness, fmt=".1f",
            on_change=lambda v: setattr(self.simulator, 'render_thickness', v)
        )
        self.sliders["brush"] = panel.add_slider(
            "Brush", 3, 80, self.brush_radius, fmt=".0f", step=1,
            on_change=lambda v: setattr(self, 'brush_radius', int(v))
        )

        # --- Engine-specific params (SHAPE section) ---
        engine_sliders = self.simulator.engine.__class__.get_slider_defs()
        if engine_sliders:
            panel.add_section("SHAPE")
            for sdef in engine_sliders:
                self.sliders[sdef["key"]] = panel.add_slider(
                    sdef["label"], sdef["min"], sdef["max"], sdef["default"],
                    fmt=sdef.get("fmt", ".3f"),
                    step=sdef.get("step"),
                    on_change=self._make_param_callback(sdef["key"])
                )

        # --- Universal flow sliders (all engines) ---
        has_any_flow = any(abs(getattr(self.simulator, '_' + fk)) > 0.001 for fk in FLOW_KEYS)
        flow_section = panel.add_collapsible_section("FLOW", expanded=has_any_flow)
        for fdef in FLOW_SLIDER_DEFS:
            self.sliders[fdef["key"]] = panel.add_slider_to(
                flow_section,
                fdef["label"], -1.0, 1.0, getattr(self.simulator, '_' + fdef["key"]),
                fmt=".2f",
                on_change=self._make_flow_callback(fdef["key"])
            )

        # --- Actions ---
        panel.add_spacer(4)
        panel.add_button("Reseed  [R]", on_click=self._on_reset)
        panel.add_spacer(4)
        panel.add_button("Clear", on_click=self._on_clear)
        panel.add_spacer(4)
        panel.add_button("Screenshot  [S]", on_click=lambda: self._save_screenshot())

        # --- Collapsible Advanced section ---
        advanced = panel.add_collapsible_section("Advanced", expanded=False)

        # Tint RGB sliders
        self.sliders["tint_r"] = panel.add_slider_to(
            advanced, "Tint R", 0.0, 2.0, self.simulator.iridescent.tint_r, fmt=".2f",
            on_change=lambda v: setattr(self.simulator.iridescent, 'tint_r', v)
        )
        self.sliders["tint_g"] = panel.add_slider_to(
            advanced, "Tint G", 0.0, 2.0, self.simulator.iridescent.tint_g, fmt=".2f",
            on_change=lambda v: setattr(self.simulator.iridescent, 'tint_g', v)
        )
        self.sliders["tint_b"] = panel.add_slider_to(
            advanced, "Tint B", 0.0, 2.0, self.simulator.iridescent.tint_b, fmt=".2f",
            on_change=lambda v: setattr(self.simulator.iridescent, 'tint_b', v)
        )

        self.panel = panel
        self._sync_sliders_from_engine()

    def _make_param_callback(self, key):
        """Create a callback that updates a single engine parameter via simulator."""
        def callback(val):
            if key in self.simulator.smoothed_params:
                # Smoothed parameter: set EMA target
                if self.simulator.param_coupler and key in ("mu", "sigma"):
                    # Lenia mu/sigma coupling
                    mu_target = self.simulator.smoothed_params["mu"].target if "mu" in self.simulator.smoothed_params else val
                    sigma_target = self.simulator.smoothed_params["sigma"].target if "sigma" in self.simulator.smoothed_params else val
                    if key == "mu":
                        mu_target = val
                    else:
                        sigma_target = val
                    coupled_mu, coupled_sigma = self.simulator.param_coupler.couple(mu_target, sigma_target)
                    self.simulator.smoothed_params["mu"].set_target(coupled_mu)
                    self.simulator.smoothed_params["sigma"].set_target(coupled_sigma)
                else:
                    self.simulator.smoothed_params[key].set_target(val)
            else:
                # Non-smoothed parameter (e.g. R for Lenia): set engine directly
                if key in ("T", "R", "num_states", "threshold"):
                    self.simulator.engine.set_params(**{key: int(val)})
                else:
                    self.simulator.engine.set_params(**{key: val})
        return callback

    def _make_flow_callback(self, key):
        """Create a callback that updates a flow parameter on the simulator."""
        def callback(val):
            setattr(self.simulator, '_' + key, val)
        return callback

    def _on_preset_select(self, idx, name):
        if idx < len(UNIFIED_ORDER):
            self.simulator.apply_preset(UNIFIED_ORDER[idx])
            self._sync_sliders_from_engine()
            # Update hue tracking
            self._hue_value = self.simulator._hue_value
            if "hue" in self.sliders:
                self.sliders["hue"].set_value(self._hue_value)
            # Rebuild panel if engine changed (different slider defs)
            self._rebuild_panel_if_needed()

    def _rebuild_panel_if_needed(self):
        """Rebuild the panel only when engine type changes (different slider defs)."""
        if self.panel is not None:
            self._build_panel()
            # Restore preset button highlight
            if self.preset_buttons and self.simulator.preset_key in UNIFIED_ORDER:
                self.preset_buttons.selected = UNIFIED_ORDER.index(self.simulator.preset_key)
                self.preset_buttons._update_active()

    def _on_speed_change(self, val):
        self.simulator.sim_speed = val

    def _on_hue_change(self, val):
        """Update hue via cosine color wheel."""
        self._hue_value = val
        self.simulator.iridescent.set_hue_offset(val)

    def _on_reset(self):
        """Reseed only — keep all current controls, shape, and flow as-is."""
        preset = get_preset(self.simulator.preset_key)
        seed_kwargs = {}
        if "density" in preset:
            seed_kwargs["density"] = preset["density"]
        self.simulator.engine.seed(preset.get("seed", "random"), **seed_kwargs)
        # GS warmup on reseed
        if self.simulator.engine_name == "gray_scott":
            for _ in range(200):
                self.simulator.engine.step()
        # Lenia warmup so pattern establishes before flow hits
        elif self.simulator.engine_name == "lenia":
            for _ in range(200):
                self.simulator.engine.step()
        self.simulator.iridescent.reset()
        self.simulator.speed_accumulator = 0.0
        # Reset LFO phase only (keep base values from current sliders)
        if self.simulator.lfo_system:
            self.simulator.lfo_system.reset_phase()
        if self.simulator.gs_lfo_system:
            self.simulator.gs_lfo_system.reset_phase()
        if self.simulator.sl_lfo_system:
            self.simulator.sl_lfo_system.reset_phase()
        if self.simulator.mnca_lfo_system:
            self.simulator.mnca_lfo_system.reset_phase()

    def _on_clear(self):
        self.simulator.engine.clear()
        self.simulator.iridescent.reset()
        self.simulator.speed_accumulator = 0.0

    def _handle_mouse(self):
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            mx, my = pygame.mouse.get_pos()
            if mx >= self.canvas_w:
                return
            sim_size = self.simulator.sim_size
            sx = int(mx * sim_size / self.canvas_w)
            sy = int(my * sim_size / self.canvas_h)
            sim_brush = int(self.brush_radius * sim_size / self.canvas_w)
            if buttons[0]:
                self.simulator.engine.add_blob(sx, sy, radius=max(sim_brush, 3), value=0.8)
            elif buttons[2]:
                self.simulator.engine.remove_blob(sx, sy, radius=max(sim_brush, 3))

    def _draw_hud(self, screen, fps):
        if not self.show_hud:
            return

        font = self.hud_font
        stats = self.simulator.engine.stats
        preset = get_preset(self.simulator.preset_key)
        engine_label = ENGINE_LABELS.get(self.simulator.engine_name, self.simulator.engine_name)

        line = (f"{engine_label} - {preset['name']}  |  Gen: {stats['generation']:,}  |  "
                f"Alive: {stats['alive_pct']:.1f}%  |  "
                f"{self.simulator.sim_size}x{self.simulator.sim_size}  |  FPS: {fps:.0f}")
        if self.paused:
            line = "[PAUSED]  " + line

        padding = 6
        bg_height = 24
        bg_surface = pygame.Surface((self.canvas_w, bg_height), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 140))
        screen.blit(bg_surface, (0, 0))

        text_surface = font.render(line, True, (210, 215, 225))
        screen.blit(text_surface, (padding + 4, padding))

    def _save_screenshot(self):
        screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "screenshots"
        )
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(screenshots_dir, f"ca_{self.simulator.preset_key}_{timestamp}.png")
        latest_path = os.path.join(screenshots_dir, "latest.png")

        # Get RGB array from simulator, convert to Surface for saving
        rgb = self.simulator.step(0.016)
        save_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
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

            # Render canvas
            screen.fill(THEME["bg"])

            if not self.paused:
                # Get the next frame from the simulator (runs full step + render)
                rgb = self.simulator.step(dt)
            else:
                # Paused: re-render without advancing simulation
                rgb = self.simulator._render_frame(dt)

            # Convert numpy rgb to pygame Surface
            sim_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
            scaled = pygame.transform.smoothscale(sim_surface, (self.canvas_w, self.canvas_h))
            screen.blit(scaled, (0, 0))

            # FPS
            frame_time = time.time() - frame_start
            self.fps_history.append(frame_time)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            avg_fps = 1.0 / max(float(np.mean(self.fps_history)), 0.001)

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
            self._save_screenshot()

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
                self.simulator.apply_preset(UNIFIED_ORDER[idx])
                self._sync_sliders_from_engine()
                self._rebuild_panel_if_needed()
                if self.preset_buttons:
                    self.preset_buttons.selected = idx
                    self.preset_buttons._update_active()
