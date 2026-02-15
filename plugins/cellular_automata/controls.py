"""
Custom UI Controls for the Cellular Automata Viewer

Minimal, dark-themed widgets drawn directly with pygame.
"""

import pygame
import math


# Theme colors
THEME = {
    "bg": (18, 18, 24),
    "panel": (25, 25, 35),
    "panel_header": (32, 32, 45),
    "track": (50, 50, 65),
    "track_fill": (80, 140, 220),
    "handle": (200, 210, 230),
    "handle_active": (255, 255, 255),
    "text": (180, 185, 195),
    "text_bright": (230, 235, 245),
    "text_dim": (100, 105, 115),
    "button": (40, 42, 55),
    "button_hover": (55, 58, 75),
    "button_active": (70, 100, 180),
    "accent": (80, 140, 220),
    "divider": (40, 40, 55),
}


class Slider:
    """Horizontal slider with label and value display."""

    def __init__(self, x, y, width, label, min_val, max_val, value,
                 fmt=".3f", step=None, on_change=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = 36
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.fmt = fmt
        self.step = step
        self.on_change = on_change
        self.dragging = False
        self.hovered = False

        # Layout
        self.track_y = self.y + 22
        self.track_h = 4
        self.handle_r = 7
        self.track_x = self.x + 8
        self.track_w = self.width - 16

    def _val_to_x(self, val):
        frac = (val - self.min_val) / (self.max_val - self.min_val)
        return self.track_x + frac * self.track_w

    def _x_to_val(self, px):
        frac = (px - self.track_x) / self.track_w
        frac = max(0, min(1, frac))
        val = self.min_val + frac * (self.max_val - self.min_val)
        if self.step:
            val = round(val / self.step) * self.step
        return val

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            hx = self._val_to_x(self.value)
            # Check if clicking on handle or track
            if (self.track_x - 4 <= mx <= self.track_x + self.track_w + 4 and
                    self.track_y - 12 <= my <= self.track_y + 12):
                self.dragging = True
                self.value = self._x_to_val(mx)
                if self.on_change:
                    self.on_change(self.value)
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos
            hx = self._val_to_x(self.value)
            self.hovered = (abs(mx - hx) < 12 and abs(my - self.track_y) < 12)

            if self.dragging:
                self.value = self._x_to_val(mx)
                if self.on_change:
                    self.on_change(self.value)
                return True

        return False

    def set_value(self, val):
        self.value = max(self.min_val, min(self.max_val, val))

    def draw(self, surface, font):
        # Label
        label_surf = font.render(self.label, True, THEME["text"])
        surface.blit(label_surf, (self.x + 8, self.y + 2))

        # Value
        val_str = f"{self.value:{self.fmt}}"
        val_surf = font.render(val_str, True, THEME["text_bright"])
        surface.blit(val_surf, (self.x + self.width - val_surf.get_width() - 8, self.y + 2))

        # Track background
        track_rect = pygame.Rect(self.track_x, self.track_y - self.track_h // 2,
                                 self.track_w, self.track_h)
        pygame.draw.rect(surface, THEME["track"], track_rect, border_radius=2)

        # Track fill
        hx = self._val_to_x(self.value)
        fill_rect = pygame.Rect(self.track_x, self.track_y - self.track_h // 2,
                                hx - self.track_x, self.track_h)
        pygame.draw.rect(surface, THEME["track_fill"], fill_rect, border_radius=2)

        # Handle
        color = THEME["handle_active"] if (self.dragging or self.hovered) else THEME["handle"]
        r = self.handle_r + (2 if self.dragging else 0)
        pygame.draw.circle(surface, color, (int(hx), self.track_y), r)

        # Inner dot
        if self.dragging:
            pygame.draw.circle(surface, THEME["track_fill"], (int(hx), self.track_y), 3)


class Button:
    """Clickable button with label."""

    def __init__(self, x, y, width, height, label, on_click=None, active=False):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.on_click = on_click
        self.active = active
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        return False

    def draw(self, surface, font):
        if self.active:
            color = THEME["button_active"]
        elif self.hovered:
            color = THEME["button_hover"]
        else:
            color = THEME["button"]

        pygame.draw.rect(surface, color, self.rect, border_radius=4)

        label_surf = font.render(self.label, True, THEME["text_bright"])
        lx = self.rect.x + (self.rect.width - label_surf.get_width()) // 2
        ly = self.rect.y + (self.rect.height - label_surf.get_height()) // 2
        surface.blit(label_surf, (lx, ly))


class ButtonRow:
    """Row of selectable buttons (like radio buttons)."""

    def __init__(self, x, y, width, labels, selected=0, on_select=None, btn_height=26):
        self.x = x
        self.y = y
        self.width = width
        self.labels = labels
        self.selected = selected
        self.on_select = on_select

        # Layout buttons in a wrapping grid
        self.buttons = []
        padding = 4
        bx = x
        by = y
        for i, label in enumerate(labels):
            # Estimate button width from label
            bw = max(len(label) * 8 + 16, 50)
            if bx + bw > x + width and bx > x:
                bx = x
                by += btn_height + padding
            btn = Button(bx, by, bw, btn_height, label)
            self.buttons.append(btn)
            bx += bw + padding

        self.total_height = by - y + btn_height
        self._update_active()

    def _update_active(self):
        for i, btn in enumerate(self.buttons):
            btn.active = (i == self.selected)

    def handle_event(self, event):
        for i, btn in enumerate(self.buttons):
            if btn.handle_event(event):
                self.selected = i
                self._update_active()
                if self.on_select:
                    self.on_select(i, self.labels[i])
                return True
            # Still handle hover
            if event.type == pygame.MOUSEMOTION:
                btn.hovered = btn.rect.collidepoint(event.pos)
        return False

    def draw(self, surface, font):
        for btn in self.buttons:
            btn.draw(surface, font)


class SectionHeader:
    """Section divider with title."""

    def __init__(self, x, y, width, title):
        self.x = x
        self.y = y
        self.width = width
        self.title = title
        self.height = 24

    def draw(self, surface, font):
        # Divider line
        pygame.draw.line(surface, THEME["divider"],
                         (self.x + 8, self.y + 8),
                         (self.x + self.width - 8, self.y + 8))
        # Title
        title_surf = font.render(self.title, True, THEME["text_dim"])
        surface.blit(title_surf, (self.x + 8, self.y + 12))


class ControlPanel:
    """
    Side panel containing all controls.
    Manages layout, events, and rendering for the control widgets.
    """

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.widgets = []  # List of (widget, type) tuples
        self.surface = pygame.Surface((width, height))
        self._cursor_y = 8  # Current vertical position for adding widgets

    def add_section(self, title):
        header = SectionHeader(0, self._cursor_y, self.width, title)
        self.widgets.append(header)
        self._cursor_y += header.height + 4

    def add_slider(self, label, min_val, max_val, value, fmt=".3f",
                   step=None, on_change=None):
        slider = Slider(0, self._cursor_y, self.width, label,
                        min_val, max_val, value, fmt, step, on_change)
        self.widgets.append(slider)
        self._cursor_y += slider.height + 6
        return slider

    def add_button_row(self, labels, selected=0, on_select=None):
        row = ButtonRow(8, self._cursor_y, self.width - 16, labels,
                        selected, on_select)
        self.widgets.append(row)
        self._cursor_y += row.total_height + 8
        return row

    def add_button(self, label, width=None, on_click=None):
        bw = width or (self.width - 16)
        btn = Button(8, self._cursor_y, bw, 28, label, on_click)
        self.widgets.append(btn)
        self._cursor_y += 36
        return btn

    def add_spacer(self, height=8):
        self._cursor_y += height

    def handle_event(self, event):
        """Process events, adjusting coordinates for panel position."""
        # Create adjusted event with local coordinates
        if hasattr(event, 'pos'):
            local_pos = (event.pos[0] - self.x, event.pos[1] - self.y)
            # Only handle if within panel bounds
            if not (0 <= local_pos[0] <= self.width and 0 <= local_pos[1] <= self.height):
                # Still need to handle mouse up for dragging sliders
                if event.type == pygame.MOUSEBUTTONUP:
                    for widget in self.widgets:
                        if hasattr(widget, 'dragging'):
                            widget.dragging = False
                return False
            # Create a modified event
            adjusted = pygame.event.Event(event.type, {
                **{k: v for k, v in event.__dict__.items() if k != 'pos'},
                'pos': local_pos
            })
        else:
            adjusted = event

        for widget in self.widgets:
            if hasattr(widget, 'handle_event'):
                if widget.handle_event(adjusted):
                    return True
        return False

    def draw(self, target_surface, font):
        """Draw the panel onto the target surface."""
        self.surface.fill(THEME["panel"])

        # Panel border
        pygame.draw.line(self.surface, THEME["divider"],
                         (0, 0), (0, self.height))

        for widget in self.widgets:
            widget.draw(self.surface, font)

        target_surface.blit(self.surface, (self.x, self.y))
