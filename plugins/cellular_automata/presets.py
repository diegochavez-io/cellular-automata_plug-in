"""
Cellular Automata Parameter Presets

Each preset defines an engine type and configuration known to produce
interesting behaviors. The "engine" field determines which CA engine
to instantiate (lenia, life, excitable, gray_scott).
"""

PRESETS = {
    # =====================================================================
    # LENIA
    # =====================================================================
    "orbium": {
        "engine": "lenia",
        "name": "Orbium",
        "description": "Gliding organism - a classic Lenia 'creature'",
        "R": 13, "T": 10, "mu": 0.15, "sigma": 0.017,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "random",
    },
    "geminium": {
        "engine": "lenia",
        "name": "Geminium",
        "description": "Self-replicating pattern that splits and multiplies",
        "R": 10, "T": 10, "mu": 0.14, "sigma": 0.014,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "blobs",
    },
    "scutium": {
        "engine": "lenia",
        "name": "Scutium",
        "description": "Shield-shaped stable structures",
        "R": 13, "T": 10, "mu": 0.21, "sigma": 0.032,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "random",
    },
    "aquarium": {
        "engine": "lenia",
        "name": "Aquarium",
        "description": "Rich ecosystem with diverse interacting structures",
        "R": 15, "T": 12, "mu": 0.16, "sigma": 0.020,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "blobs",
    },
    "mitosis": {
        "engine": "lenia",
        "name": "Mitosis",
        "description": "Blob division - organic cell-splitting behavior",
        "R": 12, "T": 10, "mu": 0.13, "sigma": 0.013,
        "kernel_peaks": [0.5], "kernel_widths": [0.12],
        "seed": "blobs",
    },
    "dual_ring": {
        "engine": "lenia",
        "name": "Dual Ring",
        "description": "Two-ring kernel producing complex interference patterns",
        "R": 18, "T": 12, "mu": 0.18, "sigma": 0.022,
        "kernel_peaks": [0.33, 0.66], "kernel_widths": [0.12, 0.12],
        "seed": "random",
    },
    "coral": {
        "engine": "lenia",
        "name": "Coral",
        "description": "Branching growth patterns like coral reefs",
        "R": 20, "T": 15, "mu": 0.12, "sigma": 0.010,
        "kernel_peaks": [0.5], "kernel_widths": [0.18],
        "seed": "blobs",
    },
    "cardiac": {
        "engine": "lenia",
        "name": "Cardiac Waves",
        "description": "Spiral wave excitation similar to cardiac tissue",
        "R": 10, "T": 8, "mu": 0.22, "sigma": 0.035,
        "kernel_peaks": [0.5], "kernel_widths": [0.20],
        "seed": "ring",
    },
    "primordial": {
        "engine": "lenia",
        "name": "Primordial Soup",
        "description": "Dense chaotic field that self-organizes over time",
        "R": 13, "T": 10, "mu": 0.15, "sigma": 0.017,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "dense",
    },

    # =====================================================================
    # GAME OF LIFE
    # =====================================================================
    "classic_life": {
        "engine": "life",
        "name": "Classic Life",
        "description": "Conway's Game of Life - the original B3/S23",
        "rule": "B3/S23", "neighborhood": "moore", "fade_rate": 0.92,
        "seed": "random", "density": 0.35,
    },
    "highlife": {
        "engine": "life",
        "name": "HighLife",
        "description": "B36/S23 - features a self-replicating pattern",
        "rule": "B36/S23", "neighborhood": "moore", "fade_rate": 0.90,
        "seed": "center", "density": 0.30,
    },
    "day_night": {
        "engine": "life",
        "name": "Day & Night",
        "description": "Symmetric rule - patterns work in positive and negative",
        "rule": "B3678/S34678", "neighborhood": "moore", "fade_rate": 0.94,
        "seed": "random", "density": 0.45,
    },
    "diamoeba": {
        "engine": "life",
        "name": "Diamoeba",
        "description": "Amoeba-like growth with diamond shapes",
        "rule": "B35678/S5678", "neighborhood": "moore", "fade_rate": 0.93,
        "seed": "random", "density": 0.50,
    },
    "seeds": {
        "engine": "life",
        "name": "Seeds",
        "description": "Explosive growth - every birth dies next step",
        "rule": "B2/S", "neighborhood": "moore", "fade_rate": 0.85,
        "seed": "sparse",
    },

    # =====================================================================
    # EXCITABLE MEDIA
    # =====================================================================
    "spiral": {
        "engine": "excitable",
        "name": "Spiral Waves",
        "description": "Classic spiral wave patterns in excitable media",
        "num_states": 8, "threshold": 2, "neighborhood": "moore",
        "seed": "random", "density": 0.15,
    },
    "turbulent": {
        "engine": "excitable",
        "name": "Turbulent",
        "description": "Many states and high threshold create turbulent dynamics",
        "num_states": 16, "threshold": 3, "neighborhood": "moore",
        "seed": "random", "density": 0.12,
    },
    "target": {
        "engine": "excitable",
        "name": "Target Waves",
        "description": "Expanding ring patterns from excitation centers",
        "num_states": 5, "threshold": 1, "neighborhood": "moore",
        "seed": "center_burst",
    },

    # =====================================================================
    # GRAY-SCOTT REACTION-DIFFUSION
    # =====================================================================
    "gs_mitosis": {
        "engine": "gray_scott",
        "name": "Mitosis",
        "description": "Spots that grow and split like dividing cells",
        "feed": 0.028, "kill": 0.062, "Du": 0.2097, "Dv": 0.105,
        "seed": "center",
    },
    "gs_coral": {
        "engine": "gray_scott",
        "name": "Coral Growth",
        "description": "Branching coral-like patterns from reaction front",
        "feed": 0.060, "kill": 0.062, "Du": 0.2097, "Dv": 0.105,
        "seed": "center",
    },
    "gs_maze": {
        "engine": "gray_scott",
        "name": "Labyrinth",
        "description": "Labyrinthine stripe patterns fill the space",
        "feed": 0.029, "kill": 0.057, "Du": 0.2097, "Dv": 0.105,
        "seed": "center",
    },
    "gs_spots": {
        "engine": "gray_scott",
        "name": "Turing Spots",
        "description": "Stable Turing spots - leopard-print pattern",
        "feed": 0.035, "kill": 0.065, "Du": 0.2097, "Dv": 0.105,
        "seed": "multi",
    },
    "gs_worms": {
        "engine": "gray_scott",
        "name": "Worms",
        "description": "Worm-like structures that crawl and branch",
        "feed": 0.078, "kill": 0.061, "Du": 0.2097, "Dv": 0.105,
        "seed": "multi",
    },
}


# Preset ordering grouped by engine
PRESET_ORDERS = {
    "lenia": [
        "orbium", "geminium", "scutium", "aquarium", "mitosis",
        "dual_ring", "coral", "cardiac", "primordial",
    ],
    "life": [
        "classic_life", "highlife", "day_night", "diamoeba", "seeds",
    ],
    "excitable": [
        "spiral", "turbulent", "target",
    ],
    "gray_scott": [
        "gs_mitosis", "gs_coral", "gs_maze", "gs_spots", "gs_worms",
    ],
}

# Flat list of all presets (for CLI compatibility)
PRESET_ORDER = []
for _keys in PRESET_ORDERS.values():
    PRESET_ORDER.extend(_keys)

ENGINE_ORDER = ["lenia", "life", "excitable", "gray_scott"]


def get_preset(name):
    """Get a preset by name. Returns None if not found."""
    return PRESETS.get(name)


def get_presets_for_engine(engine_name):
    """Return ordered list of preset keys for an engine."""
    return PRESET_ORDERS.get(engine_name, [])


def list_presets(engine=None):
    """Return list of (key, name, description) for presets.
    If engine is specified, filter to that engine only."""
    if engine:
        keys = PRESET_ORDERS.get(engine, [])
    else:
        keys = PRESET_ORDER
    return [(k, PRESETS[k]["name"], PRESETS[k]["description"])
            for k in keys]
