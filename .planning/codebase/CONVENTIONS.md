# Coding Conventions

**Analysis Date:** 2026-02-16

## Naming Patterns

**Files:**
- Lowercase with underscores: `viewer.py`, `engine_base.py`, `color_layers.py`
- Descriptive names reflecting module purpose: `lenia.py`, `gray_scott.py`, `presets.py`
- Special files: `__init__.py`, `__main__.py`, `plugin.py`

**Functions:**
- Lowercase with underscores (snake_case): `_build_containment()`, `_apply_lfo()`, `add_blob()`
- Private/internal functions prefixed with single underscore: `_build_kernel()`, `_hsv_to_rgb()`, `_update_colors()`
- Action verbs for methods: `step()`, `seed()`, `apply_feedback()`, `set_params()`, `get_params()`

**Variables:**
- Snake_case for public/local variables: `sim_size`, `brush_radius`, `lfo_phase`, `preset_key`
- Leading underscore for instance variables: `_lfo_base_mu`, `_containment`, `_kernel_fft`, `_color_matrix`
- ALL_CAPS for module-level constants: `PANEL_WIDTH`, `BASE_RES`, `ENGINE_CLASSES`, `THEME`
- Descriptive accumulator names: `speed_accumulator`, `lfo_phase`, `_mass_smooth`, `hue_time`

**Types:**
- Class names in PascalCase: `Viewer`, `Lenia`, `ColorLayerSystem`, `ExamplePipeline`, `CAEngine`, `Life`, `Excitable`, `GrayScott`
- Abstract base class prefix: `CAEngine` with `engine_name` and `engine_label` class attributes
- Dictionary keys in lowercase with underscores: `kernel_peaks`, `kernel_widths`, `master_feedback`

## Code Style

**Formatting:**
- No explicit formatter configured (appears to use implicit Python conventions)
- 4-space indentation throughout
- Blank lines between method definitions (2 blank lines between class methods in some files)
- Lines generally under 100 characters

**Linting:**
- Not detected in codebase
- No `.eslintrc`, `.flake8`, or `pyproject.toml` with lint configuration found
- Code follows PEP 8 style conventions implicitly

## Import Organization

**Order:**
1. Standard library: `import math`, `import os`, `import time`, `import sys`
2. Third-party packages: `import numpy as np`, `import pygame`, `import torch`
3. Relative imports: `from .viewer import Viewer`, `from .engine_base import CAEngine`

**Path Aliases:**
- No aliases configured
- Explicit relative imports with leading dots: `from .lenia import Lenia`
- Absolute imports for installed packages: `from example_plugin.pipeline import ExamplePipeline`

## Error Handling

**Patterns:**
- Silent defaults with `None` checks: `preset = get_preset(key)` followed by `if preset is None: return`
- Value validation with bounds: `max()`, `min()`, `np.clip()` for constraining values
- Type coercion before operations: `int(val)` for integer parameters, `float()` for stats
- Exception raising only in critical paths: No extensive try/except blocks in provided code

**Example from `viewer.py` (lines 98-99):**
```python
preset = get_preset(start_preset)
self.engine_name = preset["engine"] if preset else "lenia"
```

## Logging

**Framework:** `print()` for simple output (no formal logging system)

**Patterns:**
- Informational prints with f-strings: `print(f"Loading {model_name} on {device}...")`
- Status messages: `print(f"Screenshot saved: {path}")`
- Debug info during initialization: `print(f"  Preset: {preset}")`
- Plugin confirmation: `print("✓ Example plugin registered successfully")`

**No formal logging** - uses direct print statements for startup messages and user feedback.

## Comments

**When to Comment:**
- Module docstrings explaining purpose and features: All files start with `"""..."""` module docstring
- Class docstrings with Args/Returns for public classes: `ColorLayerSystem`, `Viewer`, engine classes
- Complex algorithm explanations: LFO dynamics (lines 392-438 in viewer.py), kernel building (lenia.py)
- Parameter descriptions: Method docstrings explain Tu, R, mu, sigma in Lenia class
- Do NOT comment obvious code (naming is self-documenting)

**JSDoc/TSDoc:**
- Not applicable (Python codebase, not TypeScript)
- Uses standard Python docstrings instead

**Example from `engine_base.py` (lines 45-49):**
```python
def add_blob(self, cx, cy, radius=15, value=0.8):
    """Paint matter at (cx, cy). Default: smooth falloff."""
    Y, X = np.ogrid[:self.size, :self.size]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    influence = np.clip(1.0 - dist / radius, 0, 1) ** 2 * value
```

## Function Design

**Size:** Methods range from 5-50 lines
- Short helper methods: `_val_to_x()` (lines 57-59), `_x_to_val()` (lines 61-67)
- Medium methods: `seed()`, `step()`, `apply_feedback()` (10-20 lines)
- Longer orchestration methods: `run()` (main loop, 80+ lines), `_apply_lfo()` (state-coupled oscillator, 60+ lines)

**Parameters:**
- Load-time parameters in `__init__()`: `num_inference_steps`, `guidance_scale`, `device`
- Runtime parameters via `**kwargs`: `strength`, `seed`, `video` - read fresh each call
- Optional parameters with defaults: `seed_type="random"`, `fmt=".3f"`, `on_change=None`

**Return Values:**
- Methods return primary computation: `step()` returns `self.world`, `composite()` returns RGB array
- Properties used for derived data: `@property def total_w(self)` calculates canvas width
- Void methods update state: `apply_feedback()`, `set_params()` modify self in-place
- Dict returns for structured results: `get_params()` returns parameter dict, `stats` property

**Example from `pipeline.py` (lines 65-120):**
```python
def __call__(self, prompt: str, **kwargs):
    """Runtime parameters MUST be read from kwargs, not from __init__."""
    video_input = kwargs.get("video", None)
    strength = kwargs.get("strength", 0.75)
    seed = kwargs.get("seed", 42)
    # ... use values fresh from kwargs each call
```

## Module Design

**Exports:**
- Main class exported implicitly: `from .viewer import Viewer` imports the Viewer class
- Engine classes in registry dict: `ENGINE_CLASSES = {"lenia": Lenia, "life": Life, ...}`
- Constants exported at module level: `PANEL_WIDTH`, `THEME`, `PRESETS`, `LAYER_DEFS`
- Plugin hook function: `register_pipelines(registry)` as entry point

**Barrel Files:**
- No barrel files (`__init__.py` is minimal, just `"""Cellular Automata - Lenia and beyond"""`)
- Direct imports preferred: `from .lenia import Lenia` not `from . import lenia`
- Plugin registration via entry point in `pyproject.toml`: `[project.entry-points."scope"]`

**Entry Points:**
- Plugin registration: `example_plugin = "example_plugin.plugin"` in pyproject.toml
- Command-line entry: `__main__.py` defines `if __name__ == "__main__": main()`
- Standalone execution: `python -m cellular_automata [preset]` loads preset and runs viewer

---

*Convention analysis: 2026-02-16*
