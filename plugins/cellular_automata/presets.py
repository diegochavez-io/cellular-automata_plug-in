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
    "amoeba": {
        "engine": "lenia",
        "name": "Lenia Glider",
        "description": "Gliding organism - a classic Lenia 'creature'",
        "R": 13, "T": 10, "mu": 0.15, "sigma": 0.017,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "random",
    },
    "coral": {
        "engine": "lenia",
        "name": "Lenia Branch",
        "description": "Branching growth patterns",
        "R": 20, "T": 15, "mu": 0.12, "sigma": 0.010,
        "kernel_peaks": [0.5], "kernel_widths": [0.18],
        "seed": "blobs",
    },
    "heartbeat": {
        "engine": "lenia",
        "name": "Lenia Spiral",
        "description": "Spiral wave excitation patterns",
        "R": 10, "T": 8, "mu": 0.22, "sigma": 0.035,
        "kernel_peaks": [0.5], "kernel_widths": [0.20],
        "seed": "ring",
    },
    "jellyfish": {
        "engine": "lenia",
        "name": "Lenia Pulse",
        "description": "Thick, pulsing, flowing mass",
        "R": 22, "T": 12, "mu": 0.22, "sigma": 0.028,
        "kernel_peaks": [0.5], "kernel_widths": [0.22],
        "seed": "blobs", "density": 0.7,
        "palette": "bioluminescent",
    },
    "lava_lamp": {
        "engine": "lenia",
        "name": "Lenia Blob",
        "description": "Slow, globular, merging/splitting blobs",
        "R": 25, "T": 18, "mu": 0.18, "sigma": 0.024,
        "kernel_peaks": [0.4, 0.8], "kernel_widths": [0.18, 0.12],
        "seed": "blobs", "density": 0.6,
        "palette": "deep_coral",
    },
    "nebula": {
        "engine": "lenia",
        "name": "Lenia Cloud",
        "description": "Multi-scale cloud-like interference",
        "R": 20, "T": 12, "mu": 0.16, "sigma": 0.020,
        "kernel_peaks": [0.3, 0.6, 0.9], "kernel_widths": [0.15, 0.12, 0.10],
        "seed": "random", "density": 0.4,
        "palette": "oil_slick",
    },
    "tide_pool": {
        "engine": "lenia",
        "name": "Lenia Wave",
        "description": "Sweeping wave-like patterns",
        "R": 18, "T": 10, "mu": 0.20, "sigma": 0.032,
        "kernel_peaks": [0.5], "kernel_widths": [0.20],
        "seed": "ring",
        "palette": "cuttlefish",
    },
    "mycelium": {
        "engine": "lenia",
        "name": "Lenia Thread",
        "description": "Thin branching networks",
        "R": 28, "T": 15, "mu": 0.11, "sigma": 0.008,
        "kernel_peaks": [0.5], "kernel_widths": [0.22],
        "seed": "blobs",
        "palette": "oil_slick",
    },

    # =====================================================================
    # GAME OF LIFE (kept in code, not shown in UI)
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
    # EXCITABLE MEDIA (kept in code, not shown in UI)
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
    # SMOOTHLIFE (continuous Life — smooth fields, never static)
    # =====================================================================
    "sl_gliders": {
        "engine": "smoothlife",
        "name": "SmoothLife Glider",
        "description": "Classic traveling blobs that glide and interact",
        "ri": 8, "ra": 24,
        "b1": 0.278, "b2": 0.365, "d1": 0.267, "d2": 0.445,
        "alpha_n": 0.028, "alpha_m": 0.147, "dt": 0.12,
        "seed": "blobs",
    },
    "sl_worms": {
        "engine": "smoothlife",
        "name": "SmoothLife Worm",
        "description": "Elongated crawling structures that wriggle",
        "ri": 10, "ra": 30,
        "b1": 0.257, "b2": 0.336, "d1": 0.267, "d2": 0.445,
        "alpha_n": 0.028, "alpha_m": 0.147, "dt": 0.10,
        "seed": "blobs",
        "palette": "deep_coral",
    },
    "sl_elastic": {
        "engine": "smoothlife",
        "name": "SmoothLife Elastic",
        "description": "Blobs connected by stretching elastic cords",
        "ri": 9, "ra": 27,
        "b1": 0.210, "b2": 0.330, "d1": 0.210, "d2": 0.500,
        "alpha_n": 0.020, "alpha_m": 0.130, "dt": 0.12,
        "seed": "blobs",
        "palette": "bioluminescent",
    },
    "sl_pulse": {
        "engine": "smoothlife",
        "name": "SmoothLife Pulse",
        "description": "Pulsing oscillating blobs that breathe in place",
        "ri": 8, "ra": 24,
        "b1": 0.270, "b2": 0.380, "d1": 0.260, "d2": 0.450,
        "alpha_n": 0.035, "alpha_m": 0.140, "dt": 0.15,
        "seed": "random",
        "palette": "cuttlefish",
    },
    "sl_chaos": {
        "engine": "smoothlife",
        "name": "SmoothLife Chaos",
        "description": "Turbulent mixing regime with constant activity",
        "ri": 7, "ra": 21,
        "b1": 0.230, "b2": 0.380, "d1": 0.190, "d2": 0.480,
        "alpha_n": 0.015, "alpha_m": 0.100, "dt": 0.12,
        "seed": "random",
        "palette": "oil_slick",
    },

    # =====================================================================
    # MNCA — Multi-Neighborhood Cellular Automata (continuous solitons)
    # =====================================================================
    "mnca_soliton": {
        "engine": "mnca",
        "name": "MNCA Soliton",
        "description": "Resilient traveling blobs that bounce and persist",
        "rings": [(0, 6), (8, 18)],
        "rules": [
            [{"low": 0.12, "high": 0.32}],
            [{"low": 0.08, "high": 0.32}],
        ],
        "delta": 0.04,
        "seed": "random",
    },
    "mnca_mitosis": {
        "engine": "mnca",
        "name": "MNCA Mitosis",
        "description": "Cells that grow, split, and divide like organisms",
        "rings": [(0, 6), (6, 12), (12, 20)],
        "rules": [
            [{"low": 0.12, "high": 0.35}],
            [{"low": 0.20, "high": 0.55}],
            [{"low": 0.06, "high": 0.25}],
        ],
        "delta": 0.035,
        "seed": "blobs",
        "palette": "bioluminescent",
    },
    "mnca_worm": {
        "engine": "mnca",
        "name": "MNCA Worm",
        "description": "Elongated worm-like structures that crawl",
        "rings": [(0, 3), (5, 10), (12, 18)],
        "rules": [
            [{"low": 0.18, "high": 0.38}],
            [{"low": 0.12, "high": 0.40}],
            [{"low": 0.04, "high": 0.18}],
        ],
        "delta": 0.045,
        "seed": "blobs",
        "palette": "deep_coral",
    },
    "mnca_hunt": {
        "engine": "mnca",
        "name": "MNCA Hunt",
        "description": "Small cells that chase and interact with each other",
        "rings": [(0, 5), (8, 16)],
        "rules": [
            [{"low": 0.15, "high": 0.35}],
            [{"low": 0.10, "high": 0.35}],
        ],
        "delta": 0.05,
        "seed": "random",
        "palette": "cuttlefish",
    },
    "mnca_coral": {
        "engine": "mnca",
        "name": "MNCA Coral",
        "description": "Branching growth patterns that spread and fill",
        "rings": [(0, 8), (12, 22)],
        "rules": [
            [{"low": 0.08, "high": 0.28}],
            [{"low": 0.05, "high": 0.25}],
        ],
        "delta": 0.03,
        "seed": "random",
        "palette": "oil_slick",
    },

    # =====================================================================
    # GRAY-SCOTT REACTION-DIFFUSION (kept in code, not shown in UI)
    # =====================================================================
    "reef": {
        "engine": "gray_scott",
        "name": "GS Spots",
        "description": "Holes and short tendrils in a contained blob",
        "feed": 0.037, "kill": 0.060, "Du": 0.21, "Dv": 0.105,
        "seed": "center",
    },
    "deep_sea": {
        "engine": "gray_scott",
        "name": "GS Dense",
        "description": "Dense holes with thick body, slow undulation",
        "feed": 0.034, "kill": 0.063, "Du": 0.21, "Dv": 0.11,
        "seed": "center",
    },
    "medusa": {
        "engine": "gray_scott",
        "name": "GS Worms",
        "description": "Sprawling long tendrils reaching outward",
        "feed": 0.022, "kill": 0.056, "Du": 0.20, "Dv": 0.13,
        "seed": "center",
    },
    "labyrinth": {
        "engine": "gray_scott",
        "name": "GS Maze",
        "description": "Maze-like winding patterns in a contained blob",
        "feed": 0.029, "kill": 0.057, "Du": 0.21, "Dv": 0.105,
        "seed": "center",
    },
    "tentacles": {
        "engine": "gray_scott",
        "name": "GS Solitons",
        "description": "Elongated soliton structures that crawl and branch",
        "feed": 0.026, "kill": 0.059, "Du": 0.21, "Dv": 0.12,
        "seed": "center",
    },

    # =====================================================================
    # CYCLIC CELLULAR AUTOMATA (kept in code, not shown in UI)
    # =====================================================================
    "whirlpool": {
        "engine": "cca",
        "name": "CCA Spiral",
        "description": "Classic spiraling vortices that never settle",
        "range_r": 1, "threshold": 1, "num_states": 14,
        "seed": "random",
    },
    "magma": {
        "engine": "cca",
        "name": "CCA Thick",
        "description": "Thick flowing wave fronts with turbulent mixing",
        "range_r": 2, "threshold": 3, "num_states": 18,
        "seed": "random",
        "palette": "deep_coral",
    },
    "aurora": {
        "engine": "cca",
        "name": "CCA Fine",
        "description": "Fine delicate spirals with many color bands",
        "range_r": 1, "threshold": 1, "num_states": 20,
        "seed": "random",
        "palette": "bioluminescent",
    },
    "vortex": {
        "engine": "cca",
        "name": "CCA Chaotic",
        "description": "Turbulent swirling patterns that constantly shift",
        "range_r": 2, "threshold": 2, "num_states": 12,
        "seed": "random",
    },
    "storm": {
        "engine": "cca",
        "name": "CCA Waves",
        "description": "Large-scale rolling fronts with deep structure",
        "range_r": 3, "threshold": 5, "num_states": 16,
        "seed": "random",
        "palette": "cuttlefish",
    },
}


# Unified preset order — shown in UI, number keys 1-9 map here
UNIFIED_ORDER = [
    # Lenia
    "amoeba", "lava_lamp", "tide_pool",
    # SmoothLife
    "sl_gliders", "sl_worms", "sl_pulse",
    # MNCA
    "mnca_mitosis", "mnca_worm",
]

# Legacy per-engine ordering (kept for compatibility)
PRESET_ORDERS = {
    "lenia": [
        "amoeba", "coral", "heartbeat", "jellyfish", "lava_lamp",
        "nebula", "tide_pool", "mycelium",
    ],
    "life": [
        "classic_life", "highlife", "day_night", "diamoeba", "seeds",
    ],
    "excitable": [
        "spiral", "turbulent", "target",
    ],
    "smoothlife": [
        "sl_gliders", "sl_worms", "sl_elastic", "sl_pulse", "sl_chaos",
    ],
    "mnca": [
        "mnca_soliton", "mnca_mitosis", "mnca_worm", "mnca_hunt", "mnca_coral",
    ],
    "gray_scott": [
        "reef", "deep_sea", "medusa", "labyrinth", "tentacles",
    ],
    "cca": [
        "whirlpool", "magma", "aurora", "vortex", "storm",
    ],
}

# Flat list of all presets (for CLI compatibility)
PRESET_ORDER = list(UNIFIED_ORDER)  # Start with unified order
for _keys in PRESET_ORDERS.values():
    for _k in _keys:
        if _k not in PRESET_ORDER:
            PRESET_ORDER.append(_k)

ENGINE_ORDER = ["lenia", "smoothlife", "mnca", "life", "excitable", "gray_scott", "cca"]


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
            for k in keys if k in PRESETS]
