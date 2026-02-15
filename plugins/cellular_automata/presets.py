"""
Lenia Parameter Presets

Each preset defines a configuration known to produce interesting behaviors.
Parameters: R (kernel radius), T (time steps), mu (growth center),
sigma (growth width), kernel_peaks, kernel_widths, seed_type.
"""

PRESETS = {
    "orbium": {
        "name": "Orbium",
        "description": "Gliding organism - a classic Lenia 'creature'",
        "R": 13,
        "T": 10,
        "mu": 0.15,
        "sigma": 0.017,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.15],
        "seed": "random",
    },
    "geminium": {
        "name": "Geminium",
        "description": "Self-replicating pattern that splits and multiplies",
        "R": 10,
        "T": 10,
        "mu": 0.14,
        "sigma": 0.014,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.15],
        "seed": "blobs",
    },
    "scutium": {
        "name": "Scutium",
        "description": "Shield-shaped stable structures",
        "R": 13,
        "T": 10,
        "mu": 0.21,
        "sigma": 0.032,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.15],
        "seed": "random",
    },
    "aquarium": {
        "name": "Aquarium",
        "description": "Rich ecosystem with diverse interacting structures",
        "R": 15,
        "T": 12,
        "mu": 0.16,
        "sigma": 0.020,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.15],
        "seed": "blobs",
    },
    "mitosis": {
        "name": "Mitosis",
        "description": "Blob division - organic cell-splitting behavior",
        "R": 12,
        "T": 10,
        "mu": 0.13,
        "sigma": 0.013,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.12],
        "seed": "blobs",
    },
    "dual_ring": {
        "name": "Dual Ring",
        "description": "Two-ring kernel producing complex interference patterns",
        "R": 18,
        "T": 12,
        "mu": 0.18,
        "sigma": 0.022,
        "kernel_peaks": [0.33, 0.66],
        "kernel_widths": [0.12, 0.12],
        "seed": "random",
    },
    "coral": {
        "name": "Coral",
        "description": "Branching growth patterns like coral reefs",
        "R": 20,
        "T": 15,
        "mu": 0.12,
        "sigma": 0.010,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.18],
        "seed": "blobs",
    },
    "cardiac": {
        "name": "Cardiac Waves",
        "description": "Spiral wave excitation similar to cardiac tissue",
        "R": 10,
        "T": 8,
        "mu": 0.22,
        "sigma": 0.035,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.20],
        "seed": "ring",
    },
    "primordial": {
        "name": "Primordial Soup",
        "description": "Dense chaotic field that self-organizes over time",
        "R": 13,
        "T": 10,
        "mu": 0.15,
        "sigma": 0.017,
        "kernel_peaks": [0.5],
        "kernel_widths": [0.15],
        "seed": "dense",
    },
}

# Ordered list for cycling through with number keys
PRESET_ORDER = [
    "orbium", "geminium", "scutium", "aquarium", "mitosis",
    "dual_ring", "coral", "cardiac", "primordial",
]


def get_preset(name):
    """Get a preset by name. Returns None if not found."""
    return PRESETS.get(name)


def list_presets():
    """Return list of (key, name, description) for all presets."""
    return [(k, PRESETS[k]["name"], PRESETS[k]["description"])
            for k in PRESET_ORDER]
