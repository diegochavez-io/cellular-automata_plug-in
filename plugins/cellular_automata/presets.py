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
        "R": 13, "T": 25, "mu": 0.15, "sigma": 0.030,
        "kernel_peaks": [0.5], "kernel_widths": [0.15],
        "seed": "blobs", "density": 0.4,
        "hue": 0.54, "brightness": 1.60, "speed": 1.56,
        "flow_swirl": 0.50, "flow_rotate": 0.35, "flow_vortex": 0.30,
        "flow_bubble": 0.20, "flow_radial": -0.15, "flow_ring": -0.95,
    },
    "coral": {
        "engine": "lenia",
        "name": "Lenia Branch",
        "description": "Branching growth patterns",
        "R": 20, "T": 15, "mu": 0.12, "sigma": 0.010,
        "kernel_peaks": [0.5], "kernel_widths": [0.18],
        "seed": "blobs",
        "flow_rotate": 0.40, "flow_swirl": 0.35, "flow_vortex": 0.25,
        "flow_radial": -0.10, "flow_bubble": 0.15,
    },
    "heartbeat": {
        "engine": "lenia",
        "name": "Lenia Spiral",
        "description": "Spiral wave excitation patterns",
        "R": 10, "T": 8, "mu": 0.22, "sigma": 0.035,
        "kernel_peaks": [0.5], "kernel_widths": [0.20],
        "seed": "ring",
        "flow_swirl": 0.45, "flow_vortex": 0.40, "flow_bubble": 0.25,
        "flow_rotate": 0.30, "flow_radial": -0.10,
    },
    "jellyfish": {
        "engine": "lenia",
        "name": "Lenia Pulse",
        "description": "Thick, pulsing, flowing mass",
        "R": 22, "T": 12, "mu": 0.22, "sigma": 0.028,
        "kernel_peaks": [0.5], "kernel_widths": [0.22],
        "seed": "blobs", "density": 0.7,
        "palette": "bioluminescent",
        "flow_bubble": 0.40, "flow_rotate": -0.30, "flow_vortex": 0.35,
        "flow_swirl": 0.25, "flow_radial": -0.15,
    },
    "lava_lamp": {
        "engine": "lenia",
        "name": "Lenia Blob",
        "description": "Slow, globular, merging/splitting blobs",
        "R": 25, "T": 25, "mu": 0.18, "sigma": 0.045,
        "kernel_peaks": [0.4, 0.8], "kernel_widths": [0.18, 0.12],
        "seed": "blobs", "density": 0.6,
        "palette": "deep_coral",
        "hue": 0.54, "brightness": 1.60,
        "flow_radial": -1.00, "flow_rotate": 0.99, "flow_swirl": 0.30,
        "flow_bubble": 0.54, "flow_vortex": 1.00,
    },
    "nebula": {
        "engine": "lenia",
        "name": "Lenia Cloud",
        "description": "Multi-scale cloud-like interference",
        "R": 20, "T": 12, "mu": 0.16, "sigma": 0.020,
        "kernel_peaks": [0.3, 0.6, 0.9], "kernel_widths": [0.15, 0.12, 0.10],
        "seed": "random", "density": 0.4,
        "palette": "oil_slick",
        "flow_rotate": 0.35, "flow_swirl": 0.30, "flow_vortex": 0.25,
        "flow_bubble": 0.15, "flow_radial": -0.10,
    },
    "tide_pool": {
        "engine": "lenia",
        "name": "Lenia Wave",
        "description": "Sweeping wave-like patterns with complex interior",
        "R": 18, "T": 25, "mu": 0.20, "sigma": 0.032,
        "kernel_peaks": [0.3, 0.7], "kernel_widths": [0.18, 0.14],
        "seed": "blobs", "density": 0.5,
        "palette": "cuttlefish",
        "hue": 0.90, "brightness": 1.50, "speed": 1.26, "thickness": 4.7,
        "flow_radial": -0.20, "flow_rotate": 0.35, "flow_swirl": -0.90,
        "flow_bubble": 0.54, "flow_ring": -0.58, "flow_vortex": -0.90,
    },
    "mycelium": {
        "engine": "lenia",
        "name": "Lenia Thread",
        "description": "Thin branching networks",
        "R": 28, "T": 15, "mu": 0.11, "sigma": 0.008,
        "kernel_peaks": [0.5], "kernel_widths": [0.22],
        "seed": "blobs",
        "palette": "oil_slick",
        "flow_rotate": 0.30, "flow_swirl": 0.25, "flow_vortex": 0.20,
        "flow_radial": -0.10, "flow_bubble": 0.10,
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
        "flow_swirl": 0.40, "flow_rotate": 0.30, "flow_vortex": 0.25,
        "flow_bubble": 0.15, "flow_radial": -0.10,
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
        "flow_vertical": 0.20, "flow_swirl": 0.30, "flow_rotate": 0.25,
        "flow_vortex": 0.20,
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
        "flow_rotate": 0.35, "flow_bubble": 0.25, "flow_vortex": 0.20,
        "flow_swirl": 0.30,
    },
    "sl_pulse": {
        "engine": "smoothlife",
        "name": "SmoothLife Pulse",
        "description": "Pulsing oscillating blobs that breathe in place",
        "ri": 8, "ra": 24,
        "b1": 0.377, "b2": 0.474, "d1": 0.260, "d2": 0.436,
        "alpha_n": 0.035, "alpha_m": 0.140, "dt": 0.15,
        "seed": "random",
        "palette": "cuttlefish",
        "hue": 0.25, "brightness": 1.45,
        "flow_radial": 0.54, "flow_rotate": -0.94, "flow_swirl": 0.98,
        "flow_bubble": 0.88, "flow_ring": 0.89, "flow_vortex": 0.15,
        "flow_vertical": -0.15,
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
        "flow_vortex": 0.40, "flow_swirl": 0.35, "flow_rotate": 0.25,
        "flow_bubble": 0.15,
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
        "flow_rotate": 0.35, "flow_swirl": 0.30, "flow_vortex": 0.25,
        "flow_bubble": 0.15, "flow_radial": -0.10,
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
        "flow_bubble": 0.40, "flow_rotate": 0.30, "flow_swirl": 0.25,
        "flow_vortex": 0.20, "flow_radial": -0.15,
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
        "flow_rotate": 0.40, "flow_swirl": 0.30, "flow_vortex": 0.25,
        "flow_bubble": 0.15, "flow_radial": -0.10,
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
        "flow_vortex": 0.35, "flow_rotate": 0.30, "flow_swirl": 0.25,
        "flow_bubble": 0.15, "flow_radial": -0.10,
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
        "flow_rotate": 0.35, "flow_bubble": 0.25, "flow_swirl": 0.30,
        "flow_vortex": 0.20, "flow_radial": -0.10,
    },

    # =====================================================================
    # GRAY-SCOTT REACTION-DIFFUSION (with flow field advection)
    # =====================================================================
    "reef": {
        "engine": "gray_scott",
        "name": "GS Coral",
        "description": "Slowly spinning spots that never settle",
        "feed": 0.037, "kill": 0.060, "Du": 0.21, "Dv": 0.105,
        "seed": "dense",
        "flow_rotate": 0.35, "flow_radial": 0.15,
    },
    "deep_sea": {
        "engine": "gray_scott",
        "name": "GS Drift",
        "description": "Ocean current drift with gentle swirl",
        "feed": 0.034, "kill": 0.063, "Du": 0.21, "Dv": 0.11,
        "seed": "dense",
        "flow_vertical": 0.45, "flow_swirl": 0.20,
    },
    "medusa": {
        "engine": "gray_scott",
        "name": "GS Medusa",
        "description": "Pulsing jellyfish expansion with slow counter-spin",
        "feed": 0.022, "kill": 0.056, "Du": 0.20, "Dv": 0.13,
        "seed": "dense",
        "flow_bubble": 0.50, "flow_rotate": -0.15,
    },
    "labyrinth": {
        "engine": "gray_scott",
        "name": "GS Labyrinth",
        "description": "Spiraling mazes with concentric rings",
        "feed": 0.029, "kill": 0.057, "Du": 0.21, "Dv": 0.105,
        "seed": "dense",
        "flow_swirl": 0.60, "flow_ring": 0.25,
    },
    "tentacles": {
        "engine": "gray_scott",
        "name": "GS Vortex",
        "description": "Whirlpool solitons spiraling inward",
        "feed": 0.026, "kill": 0.059, "Du": 0.21, "Dv": 0.12,
        "seed": "dense",
        "flow_vortex": 0.40, "flow_radial": -0.12,
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
    # Gray-Scott (flow advection)
    "labyrinth", "tentacles", "medusa",
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
