"""
Interactive Pygame Viewer for Cellular Automata

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

from .lenia import Lenia
from .colormaps import get_colormap, apply_colormap, COLORMAP_ORDER
from .presets import PRESETS, PRESET_ORDER, get_preset
from .controls import ControlPanel, THEME


PANEL_WIDTH = 300
BASE_RES = 512  # Presets are tuned for this resolution


class Viewer:
    def __init__(self, width=900, height=900, sim_size=1024, start_preset="orbium"):
        self.canvas_w = width
        self.canvas_h = height
        self.panel_visible = True
        self.sim_size = sim_size
        self.res_scale = sim_size / BASE_RES  # Scale kernel params proportionally
        self.running = True
        self.paused = False
        self.show_hud = True
        self.fullscreen = False
        self.steps_per_frame = 1
        self.brush_radius = 20
        self.fps_history = []

        # Trail / persistence buffer for smoky effect
        self.trail_buffer = None
        self.trail_decay = 0.0  # 0 = off, higher = longer trails (0-0.98)

        # Colormap state
        self.colormap_idx = 0
        self.colormap_name = COLORMAP_ORDER[0]
        self.lut = get_colormap(self.colormap_name)

        # Preset state
        self.preset_idx = PRESET_ORDER.index(start_preset) if start_preset in PRESET_ORDER else 0
        self.preset_key = PRESET_ORDER[self.preset_idx]

        # Control panel (built after pygame.init in run())
        self.panel = None
        self.sliders = {}
        self.preset_buttons = None
        self.cmap_buttons = None

        # Create Lenia engine at high resolution
        self.lenia = Lenia(size=sim_size)
        self._apply_preset(self.preset_key)

    @property
    def total_w(self):
        return self.canvas_w + (PANEL_WIDTH if self.panel_visible else 0)

    def _scale_R(self, base_R):
        """Scale kernel radius for current sim resolution."""
        return max(5, int(base_R * self.res_scale))

    def _apply_preset(self, key):
        """Apply a preset and reseed, scaling R for resolution."""
        preset = get_preset(key)
        if preset is None:
            return
        self.preset_key = key
        self.preset_idx = PRESET_ORDER.index(key) if key in PRESET_ORDER else 0
        self.lenia.set_params(
            R=self._scale_R(preset["R"]),
            T=preset["T"],
            mu=preset["mu"],
            sigma=preset["sigma"],
            kernel_peaks=preset.get("kernel_peaks"),
            kernel_widths=preset.get("kernel_widths"),
        )
        self._seed_from_preset(preset)
        self._reset_trail()
        self._sync_sliders_from_engine()

    def _seed_from_preset(self, preset):
        """Seed the world based on preset configuration."""
        seed_type = preset.get("seed", "random")
        if seed_type == "blobs":
            self.lenia.seed_multiple_blobs()
        elif seed_type == "ring":
            self.lenia.seed_ring()
        elif seed_type == "dense":
            self.lenia.seed_random(density=0.8, radius=self.sim_size // 3)
        else:
            self.lenia.seed_random()

    def _reset_trail(self):
        """Reset the trail persistence buffer."""
        self.trail_buffer = np.zeros((self.sim_size, self.sim_size), dtype=np.float64)

    def _update_trail(self):
        """Blend current world into trail buffer."""
        if self.trail_buffer is None:
            self._reset_trail()
        if self.trail_decay <= 0.001:
            return  # Trails disabled
        # Exponential moving average: trail keeps old info, blends in new
        alpha = 1.0 - self.trail_decay
        self.trail_buffer = self.trail_buffer * self.trail_decay + self.lenia.world * alpha

    def _get_display_field(self):
        """Get the field to render - either raw world or world + trails."""
        if self.trail_decay <= 0.001:
            return self.lenia.world
        # Combine: show the brighter of current world and trail
        return np.maximum(self.lenia.world, self.trail_buffer)

    def _sync_sliders_from_engine(self):
        """Update slider positions to match current engine state."""
        if not self.sliders:
            return
        self.sliders["mu"].set_value(self.lenia.mu)
        self.sliders["sigma"].set_value(self.lenia.sigma)
        self.sliders["T"].set_value(self.lenia.T)
        self.sliders["R"].set_value(self.lenia.R)
        self.sliders["speed"].set_value(self.steps_per_frame)
        self.sliders["brush"].set_value(self.brush_radius)
        self.sliders["trails"].set_value(self.trail_decay)

    def _build_panel(self):
        """Build the control panel with all widgets."""
        panel = ControlPanel(self.canvas_w, 0, PANEL_WIDTH, self.canvas_h)

        # --- Presets ---
        panel.add_section("PRESETS")
        preset_names = [PRESETS[k]["name"] for k in PRESET_ORDER]
        self.preset_buttons = panel.add_button_row(
            preset_names, selected=self.preset_idx,
            on_select=self._on_preset_select
        )

        # --- Growth Parameters ---
        panel.add_section("GROWTH FUNCTION")
        self.sliders["mu"] = panel.add_slider(
            "mu (center)", 0.01, 0.40, self.lenia.mu, fmt=".4f",
            on_change=lambda v: setattr(self.lenia, 'mu', v)
        )
        self.sliders["sigma"] = panel.add_slider(
            "sigma (width)", 0.002, 0.08, self.lenia.sigma, fmt=".4f",
            on_change=lambda v: setattr(self.lenia, 'sigma', v)
        )

        # --- Kernel ---
        panel.add_section("KERNEL")
        self.sliders["R"] = panel.add_slider(
            "Radius (R)", 5, 60, self.lenia.R, fmt=".0f", step=1,
            on_change=self._on_R_change
        )
        self.sliders["T"] = panel.add_slider(
            "Time res (T)", 1, 30, self.lenia.T, fmt=".0f", step=1,
            on_change=self._on_T_change
        )

        # --- Rendering ---
        panel.add_section("RENDERING")
        self.sliders["trails"] = panel.add_slider(
            "Trail persist", 0.0, 0.98, self.trail_decay, fmt=".2f",
            on_change=self._on_trail_change
        )

        # --- Simulation ---
        panel.add_section("SIMULATION")
        self.sliders["speed"] = panel.add_slider(
            "Steps/frame", 1, 20, self.steps_per_frame, fmt=".0f", step=1,
            on_change=lambda v: setattr(self, 'steps_per_frame', int(v))
        )
        self.sliders["brush"] = panel.add_slider(
            "Brush size", 3, 80, self.brush_radius, fmt=".0f", step=1,
            on_change=lambda v: setattr(self, 'brush_radius', int(v))
        )

        # --- Colormap ---
        panel.add_section("COLORMAP")
        cmap_names = [n.replace("_", " ").title() for n in COLORMAP_ORDER]
        self.cmap_buttons = panel.add_button_row(
            cmap_names, selected=self.colormap_idx,
            on_select=self._on_colormap_select
        )

        # --- Actions ---
        panel.add_section("ACTIONS")
        panel.add_button("Reset / Reseed  [R]", on_click=self._on_reset)
        panel.add_spacer(4)
        panel.add_button("Clear World", on_click=self._on_clear)
        panel.add_spacer(4)
        panel.add_button("Screenshot  [S]", on_click=lambda: self._save_screenshot(None))

        self.panel = panel

    def _on_preset_select(self, idx, name):
        key = PRESET_ORDER[idx]
        self._apply_preset(key)

    def _on_colormap_select(self, idx, name):
        self.colormap_idx = idx
        self.colormap_name = COLORMAP_ORDER[idx]
        self.lut = get_colormap(self.colormap_name)

    def _on_R_change(self, val):
        self.lenia.set_params(R=int(val))

    def _on_T_change(self, val):
        self.lenia.set_params(T=int(val))

    def _on_trail_change(self, val):
        self.trail_decay = val
        if val <= 0.001:
            self._reset_trail()

    def _on_reset(self):
        preset = get_preset(self.preset_key)
        self._seed_from_preset(preset)
        self._reset_trail()

    def _on_clear(self):
        self.lenia.clear()
        self._reset_trail()

    def _render_frame(self):
        """Render current world state to a pygame surface."""
        field = self._get_display_field()
        rgb = apply_colormap(field, self.lut)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
        return surface

    def _draw_hud(self, screen, fps):
        """Draw heads-up display over the canvas area."""
        if not self.show_hud:
            return

        font = self.hud_font
        stats = self.lenia.stats
        preset = get_preset(self.preset_key)

        line = (f"{preset['name']}  |  Gen: {stats['generation']:,}  |  "
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
        """Handle mouse painting/erasing on the canvas area."""
        buttons = pygame.mouse.get_pressed()
        if buttons[0] or buttons[2]:
            mx, my = pygame.mouse.get_pos()
            if mx >= self.canvas_w:
                return
            sx = int(mx * self.sim_size / self.canvas_w)
            sy = int(my * self.sim_size / self.canvas_h)
            sim_brush = int(self.brush_radius * self.sim_size / self.canvas_w)
            if buttons[0]:
                self.lenia.add_blob(sx, sy, radius=max(sim_brush, 3), value=0.8)
            elif buttons[2]:
                self.lenia.remove_blob(sx, sy, radius=max(sim_brush, 3))

    def _save_screenshot(self, screen):
        """Save current frame as PNG at full sim resolution."""
        screenshots_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "screenshots"
        )
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(screenshots_dir, f"ca_{self.preset_key}_{timestamp}.png")

        field = self._get_display_field()
        rgb = apply_colormap(field, self.lut)
        save_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
        pygame.image.save(save_surface, path)
        print(f"Screenshot saved: {path}")

    def run(self):
        """Main viewer loop."""
        pygame.init()

        screen = pygame.display.set_mode((self.total_w, self.canvas_h), pygame.RESIZABLE)
        pygame.display.set_caption("Cellular Automata - Lenia")
        clock = pygame.time.Clock()

        self.hud_font = pygame.font.SysFont("menlo", 13)
        self.panel_font = pygame.font.SysFont("menlo", 12)

        self._build_panel()

        while self.running:
            frame_start = time.time()

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

            # Simulation
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.lenia.step()
                    self._update_trail()

            # Render canvas
            screen.fill(THEME["bg"])
            sim_surface = self._render_frame()
            # Smooth upscale for anti-aliased display
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

        # Preset selection (1-9)
        elif pygame.K_1 <= key <= pygame.K_9:
            idx = key - pygame.K_1
            if idx < len(PRESET_ORDER):
                self._apply_preset(PRESET_ORDER[idx])
                if self.preset_buttons:
                    self.preset_buttons.selected = idx
                    self.preset_buttons._update_active()

        # Colormap cycling
        elif key == pygame.K_c:
            new_idx = (self.colormap_idx + 1) % len(COLORMAP_ORDER)
            self._on_colormap_select(new_idx, COLORMAP_ORDER[new_idx])
            if self.cmap_buttons:
                self.cmap_buttons.selected = new_idx
                self.cmap_buttons._update_active()
