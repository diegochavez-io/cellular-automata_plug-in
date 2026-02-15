"""
Custom Colormaps for Cellular Automata Visualization

Maps float values [0, 1] to RGB colors. Each colormap is a (256, 3)
uint8 array that serves as a lookup table.
"""

import numpy as np


def _interpolate_colors(stops, n=256):
    """
    Build a colormap by interpolating between color stops.

    Args:
        stops: List of (position, (r, g, b)) where position is [0, 1]
        n: Number of entries in the LUT
    """
    lut = np.zeros((n, 3), dtype=np.uint8)
    positions = [s[0] for s in stops]
    colors = [s[1] for s in stops]

    for i in range(n):
        t = i / (n - 1)
        # Find surrounding stops
        for j in range(len(positions) - 1):
            if positions[j] <= t <= positions[j + 1]:
                # Interpolate between stops[j] and stops[j+1]
                span = positions[j + 1] - positions[j]
                if span == 0:
                    frac = 0
                else:
                    frac = (t - positions[j]) / span
                # Smooth interpolation
                frac = frac * frac * (3 - 2 * frac)  # smoothstep
                for c in range(3):
                    lut[i, c] = int(colors[j][c] + frac * (colors[j+1][c] - colors[j][c]))
                break
    return lut


# --- Colormap Definitions ---

def neon_bio():
    """Neon biology - dark bg, green halos, red/pink cores.
    Inspired by bioluminescent organisms."""
    return _interpolate_colors([
        (0.00, (5, 2, 8)),        # near black, slight purple
        (0.05, (40, 15, 5)),      # dark brown
        (0.15, (70, 35, 10)),     # warm brown
        (0.30, (20, 180, 80)),    # green halo
        (0.45, (30, 220, 140)),   # bright cyan-green
        (0.55, (140, 40, 180)),   # purple transition
        (0.70, (220, 25, 60)),    # bright red
        (0.85, (255, 50, 90)),    # hot pink
        (1.00, (255, 130, 160)),  # soft pink peak
    ])


def smoke():
    """White smoke on black - ethereal wisp aesthetic."""
    return _interpolate_colors([
        (0.00, (0, 0, 0)),
        (0.20, (8, 10, 15)),
        (0.40, (40, 50, 60)),
        (0.60, (110, 120, 135)),
        (0.80, (190, 200, 210)),
        (1.00, (255, 255, 255)),
    ])


def plasma():
    """Purple-pink-orange-yellow plasma gradient."""
    return _interpolate_colors([
        (0.00, (10, 0, 20)),
        (0.25, (80, 10, 130)),
        (0.50, (190, 30, 100)),
        (0.75, (240, 130, 30)),
        (1.00, (245, 240, 80)),
    ])


def fire():
    """Black through red to yellow-white fire."""
    return _interpolate_colors([
        (0.00, (0, 0, 0)),
        (0.20, (60, 5, 0)),
        (0.40, (180, 30, 0)),
        (0.60, (240, 100, 10)),
        (0.80, (255, 200, 50)),
        (1.00, (255, 255, 200)),
    ])


def ocean():
    """Deep blue to cyan to white ocean depths."""
    return _interpolate_colors([
        (0.00, (0, 2, 15)),
        (0.25, (5, 20, 80)),
        (0.50, (10, 80, 160)),
        (0.75, (40, 180, 220)),
        (1.00, (200, 250, 255)),
    ])


def electric():
    """Electric blue-purple-white with high contrast."""
    return _interpolate_colors([
        (0.00, (0, 0, 0)),
        (0.15, (10, 0, 40)),
        (0.35, (30, 20, 180)),
        (0.55, (100, 50, 255)),
        (0.75, (200, 150, 255)),
        (1.00, (255, 255, 255)),
    ])


def moss():
    """Dark earth tones to vibrant green - organic growth."""
    return _interpolate_colors([
        (0.00, (5, 5, 2)),
        (0.20, (20, 30, 10)),
        (0.40, (40, 80, 20)),
        (0.60, (60, 160, 40)),
        (0.80, (100, 220, 80)),
        (1.00, (180, 255, 150)),
    ])


def thermal():
    """Thermal camera look - blue cold to red hot."""
    return _interpolate_colors([
        (0.00, (0, 0, 20)),
        (0.20, (0, 0, 120)),
        (0.40, (30, 80, 180)),
        (0.50, (60, 180, 80)),
        (0.60, (200, 200, 30)),
        (0.80, (240, 80, 0)),
        (1.00, (255, 255, 255)),
    ])


# Registry of all colormaps
COLORMAPS = {
    "neon_bio": neon_bio,
    "smoke": smoke,
    "plasma": plasma,
    "fire": fire,
    "ocean": ocean,
    "electric": electric,
    "moss": moss,
    "thermal": thermal,
}

COLORMAP_ORDER = list(COLORMAPS.keys())


def get_colormap(name):
    """Get a colormap LUT (256, 3) uint8 array by name."""
    return COLORMAPS[name]()


def apply_colormap(field, lut):
    """
    Apply a colormap LUT to a 2D float field.

    Args:
        field: 2D numpy array with values in [0, 1]
        lut: (256, 3) uint8 colormap lookup table

    Returns:
        (H, W, 3) uint8 RGB image
    """
    indices = (np.clip(field, 0, 1) * 255).astype(np.uint8)
    return lut[indices]
