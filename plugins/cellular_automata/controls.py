"""
Custom UI Controls for the Cellular Automata Viewer

Minimal, dark-themed widgets drawn directly with pygame.
"""

import pygame
import math
import time


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
        self.height = 30
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.default_value = value  # Store initial value for double-click reset
        self.fmt = fmt
        self.step = step
        self.on_change = on_change
        self.dragging = False
        self.hovered = False
        self._last_click_time = 0  # For double-click detection

        # Layout
        self.track_y = self.y + 20
        self.track_h = 4
        self.handle_r = 6
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

                current_time = time.time()
                # Double-click detection (within 300ms)
                if current_time - self._last_click_time < 0.3:
                    # Double-click: reset to default
                    self.value = self.default_value
                    if self.on_change:
                        self.on_change(self.value)
                    self._last_click_time = current_time
                    return True

                # Single click: start dragging
                self._last_click_time = current_time
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


class ColorSlider(Slider):
    """Slider with a colored swatch indicator next to the label."""

    def __init__(self, x, y, width, label, min_val, max_val, value,
                 swatch_color=(255, 255, 255), fmt=".2f", step=None,
                 on_change=None):
        super().__init__(x, y, width, label, min_val, max_val, value,
                         fmt, step, on_change)
        self.swatch_color = swatch_color

    def draw(self, surface, font):
        # Draw colored swatch square before label
        swatch_size = 10
        sx = self.x + 8
        sy = self.y + 4
        pygame.draw.rect(surface, self.swatch_color,
                         (sx, sy, swatch_size, swatch_size),
                         border_radius=2)

        # Shift label right to make room for swatch
        old_x = self.x
        self.x += swatch_size + 6
        super().draw(surface, font)
        self.x = old_x


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


class CollapsibleSection:
    """Collapsible section with a clickable header that toggles child visibility."""

    def __init__(self, x, y, width, title, expanded=False):
        self.x = x
        self.y = y
        self.width = width
        self.title = title
        self.expanded = expanded
        self.header_height = 24
        self.children = []
        self._total_children_height = 0

    @property
    def height(self):
        if self.expanded:
            return self.header_height + self._total_children_height
        return self.header_height

    def add_child(self, widget):
        self.children.append(widget)
        # Track total height of children
        h = getattr(widget, 'height', 0)
        if isinstance(widget, Slider):
            h = widget.height + 3
        elif isinstance(widget, ButtonRow):
            h = widget.total_height + 6
        elif isinstance(widget, SectionHeader):
            h = widget.height + 3
        self._total_children_height += h

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            # Check header click
            if (self.x <= mx <= self.x + self.width and
                    self.y <= my <= self.y + self.header_height):
                self.expanded = not self.expanded
                return True

        # Forward events to children only when expanded
        if self.expanded:
            for child in self.children:
                if hasattr(child, 'handle_event'):
                    if child.handle_event(event):
                        return True
        return False

    def draw(self, surface, font):
        # Draw header with toggle indicator
        arrow = "v " if self.expanded else "> "
        header_text = arrow + self.title
        # Divider line
        pygame.draw.line(surface, THEME["divider"],
                         (self.x + 8, self.y + 8),
                         (self.x + self.width - 8, self.y + 8))
        title_surf = font.render(header_text, True, THEME["text_dim"])
        surface.blit(title_surf, (self.x + 8, self.y + 12))

        # Draw children only when expanded
        if self.expanded:
            for child in self.children:
                child.draw(surface, font)


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
        self.scroll_offset = 0
        self._content_height = height

    def add_section(self, title):
        header = SectionHeader(0, self._cursor_y, self.width, title)
        self.widgets.append(header)
        self._cursor_y += header.height + 2

    def add_slider(self, label, min_val, max_val, value, fmt=".3f",
                   step=None, on_change=None):
        slider = Slider(0, self._cursor_y, self.width, label,
                        min_val, max_val, value, fmt, step, on_change)
        self.widgets.append(slider)
        self._cursor_y += slider.height + 3
        return slider

    def add_color_slider(self, label, min_val, max_val, value,
                         swatch_color=(255, 255, 255), fmt=".2f",
                         step=None, on_change=None):
        slider = ColorSlider(0, self._cursor_y, self.width, label,
                              min_val, max_val, value, swatch_color,
                              fmt, step, on_change)
        self.widgets.append(slider)
        self._cursor_y += slider.height + 3
        return slider

    def add_button_row(self, labels, selected=0, on_select=None):
        row = ButtonRow(8, self._cursor_y, self.width - 16, labels,
                        selected, on_select)
        self.widgets.append(row)
        self._cursor_y += row.total_height + 4
        return row

    def add_button(self, label, width=None, on_click=None):
        bw = width or (self.width - 16)
        btn = Button(8, self._cursor_y, bw, 28, label, on_click)
        self.widgets.append(btn)
        self._cursor_y += 36
        return btn

    def add_collapsible_section(self, title, expanded=False):
        """Add a collapsible section. Returns it so caller can add children."""
        section = CollapsibleSection(0, self._cursor_y, self.width, title, expanded)
        self.widgets.append(section)
        self._cursor_y += section.header_height
        return section

    def add_slider_to(self, section, label, min_val, max_val, value, fmt=".3f",
                      step=None, on_change=None):
        """Add a slider as a child of a collapsible section."""
        slider = Slider(0, self._cursor_y, self.width, label,
                        min_val, max_val, value, fmt, step, on_change)
        section.add_child(slider)
        self._cursor_y += slider.height + 3
        return slider

    def add_spacer(self, height=8):
        self._cursor_y += height

    def _scroll(self, amount):
        """Scroll panel by amount (positive = down)."""
        self._content_height = max(self.height, self._cursor_y + 8)
        self.scroll_offset += amount
        max_scroll = max(0, self._content_height - self.height)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

    def _mouse_over_panel(self):
        mx, my = pygame.mouse.get_pos()
        return (self.x <= mx <= self.x + self.width and
                self.y <= my <= self.y + self.height)

    def handle_event(self, event):
        """Process events, adjusting coordinates for panel position."""
        # Mouse wheel scrolling (MOUSEWHEEL event)
        # Use precise_y for macOS trackpad (event.y rounds to 0 for small scrolls)
        if event.type == pygame.MOUSEWHEEL:
            if self._mouse_over_panel():
                scroll_y = getattr(event, 'precise_y', event.y)
                if scroll_y == 0:
                    scroll_y = event.y
                self._scroll(-scroll_y * 20)
                return True
            return False

        # Mouse wheel scrolling (button 4/5 fallback for some platforms)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            if self._mouse_over_panel():
                self._scroll(-30 if event.button == 4 else 30)
                return True
            return False

        # Create adjusted event with local coordinates (offset by scroll)
        if hasattr(event, 'pos'):
            screen_local = (event.pos[0] - self.x, event.pos[1] - self.y)
            # Only handle if within visible panel bounds
            if not (0 <= screen_local[0] <= self.width and 0 <= screen_local[1] <= self.height):
                # Still need to handle mouse up for dragging sliders
                if event.type == pygame.MOUSEBUTTONUP:
                    for widget in self.widgets:
                        if hasattr(widget, 'dragging'):
                            widget.dragging = False
                return False
            # Offset y by scroll position so widgets get content-space coords
            local_pos = (screen_local[0], screen_local[1] + self.scroll_offset)
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
        """Draw the panel onto the target surface with scroll support."""
        # Ensure content surface is large enough
        self._content_height = max(self.height, self._cursor_y + 8)
        needed_h = self._content_height
        if self.surface.get_height() < needed_h or self.surface.get_width() != self.width:
            self.surface = pygame.Surface((self.width, needed_h))

        self.surface.fill(THEME["panel"])

        # Panel border
        pygame.draw.line(self.surface, THEME["divider"],
                         (0, 0), (0, needed_h))

        for widget in self.widgets:
            widget.draw(self.surface, font)

        # Blit visible portion (scrolled)
        max_scroll = max(0, self._content_height - self.height)
        self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))
        target_surface.blit(self.surface, (self.x, self.y),
                            (0, self.scroll_offset, self.width, self.height))

        # Scrollbar indicator (only when content overflows)
        if max_scroll > 0:
            bar_h = max(20, int(self.height * self.height / self._content_height))
            bar_y = int(self.scroll_offset / max_scroll * (self.height - bar_h))
            bar_rect = pygame.Rect(self.x + self.width - 4, self.y + bar_y, 3, bar_h)
            pygame.draw.rect(target_surface, THEME["text_dim"], bar_rect, border_radius=1)
