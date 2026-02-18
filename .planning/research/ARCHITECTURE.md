# Architecture Research: Scope Plugin + GPU Deployment

**Domain:** Scope plugin wrapper for cellular automata video source, with CuPy GPU acceleration and RunPod deployment
**Researched:** 2026-02-17
**Confidence:** HIGH (Scope plugin API confirmed from example_plugin, code structure verified from codebase, CuPy API patterns from official docs)

---

## System Overview

```
LOCAL (pygame mode)                 SCOPE (headless mode)
────────────────────               ────────────────────────────
pygame event loop                   Scope scheduler
    │                                   │
    ↓                                   ↓
Viewer.__init__()                   CAPipeline.__init__()
    │                                   │
    ↓                                   ↓
CASimulator (NEW — extracted)  ←──  CASimulator (shared core)
    │   engine.step()                   │   engine.step()
    │   _advect()                       │   _advect()
    │   _manage_coverage()              │   _manage_coverage()
    │   iridescent.render()             │   iridescent.render()
    │                                   │
    ↓                                   ↓
pygame surface → display        THWC tensor [0,1] → Scope
```

The key insight: extract a `CASimulator` class from `viewer.py` that holds all simulation logic without any pygame dependency. Both the pygame `Viewer` and the Scope `CAPipeline` delegate to it.

---

## Component Boundaries

### Current State (Phase 1 — before refactor)

| File | Responsibility | pygame dep? | Problem for Scope |
|------|---------------|-------------|-------------------|
| `viewer.py` | Sim loop + display + UI event handling | YES | Monolith, can't use headless |
| `iridescent.py` | Cosine palette color pipeline | NO | Ready to use as-is |
| `engine_base.py` | CAEngine abstract base | NO | Ready to use as-is |
| `lenia.py` | Lenia FFT engine | NO | Ready to use as-is |
| `smoothlife.py` | SmoothLife engine | NO | Ready to use as-is |
| `mnca.py` | MNCA ring-kernel engine | NO | Ready to use as-is |
| `gray_scott.py` | GS reaction-diffusion | NO | Ready to use as-is |
| `smoothing.py` | EMA params, LFO systems | NO | Ready to use as-is |
| `presets.py` | Preset definitions | NO | Ready to use as-is |
| `controls.py` | pygame UI widgets | YES (100%) | Drop entirely for Scope |

### Target State (Phase 2 — after refactor)

| File | Responsibility | New/Modified |
|------|---------------|--------------|
| `simulator.py` | Pure simulation logic (extracted from viewer.py) | NEW |
| `viewer.py` | pygame display wrapper that delegates to CASimulator | MODIFIED (slim) |
| `plugin.py` | Scope plugin registration (`register_pipelines`) | NEW |
| `pipeline.py` | Scope pipeline class that delegates to CASimulator | NEW |
| `numpy_backend.py` | numpy/scipy wrappers for CPU ops | NEW (Phase 3) |
| `cupy_backend.py` | CuPy GPU equivalents | NEW (Phase 3) |
| All engine files | Unchanged — pure numpy, engine-agnostic | UNCHANGED |
| `iridescent.py` | Add `render_float()` returning [0,1] instead of uint8 | MINOR |
| `presets.py` | Unchanged | UNCHANGED |

---

## What to Extract from viewer.py

`viewer.py` currently mixes two concerns. The extraction boundary is clear:

**Stays in Viewer (pygame display):**
- `pygame.init()`, `pygame.display.set_mode()`
- `_build_panel()`, `_sync_sliders_from_engine()`
- `_handle_keydown()`, `_handle_mouse()`
- `_draw_hud()`, `_save_screenshot()`
- `run()` loop with `clock.tick(60)` and `pygame.display.flip()`

**Moves to CASimulator (pure logic):**
- All LFO system classes: `LeniaLFOSystem`, `GrayScottLFOSystem`, `SmoothLifeLFOSystem`, `MNCALFOSystem`, `SinusoidalLFO`
- `_build_containment()`, `_build_noise_mask()`, `_build_stir_field()`
- `_build_color_offset()`, `_build_flow_fields()`
- `_advect()`, `_fast_noise()`, `_dilate_world()`, `_blur_world()`
- `_apply_preset()`, `_create_engine()`
- `_drop_center_seed()`, `_drop_seed_cluster()`
- `_render_gs_emboss()` and `_render_frame()` → return `np.ndarray` not `pygame.Surface`
- `_apply_bloom()`
- Per-frame sim logic: smoothed params, LFO update, `engine.step()`, advection, containment, noise/stir
- Coverage management: `_prev_mass`, `_stagnant_frames`, `_nucleation_counter`
- The `speed_accumulator` fractional speed system

**New CASimulator interface:**
```python
class CASimulator:
    def __init__(self, preset_key="coral", sim_size=None):
        """Initialize with preset. sim_size=None uses preset default."""
        ...

    def apply_preset(self, key: str):
        """Switch to a named preset (engine + params + seed)."""
        ...

    def set_runtime_params(self, **kwargs):
        """Accept Scope kwargs: preset, speed, hue, brightness, flow_*, thickness."""
        ...

    def step(self, dt: float) -> np.ndarray:
        """Advance simulation by dt seconds. Returns RGB (H, W, 3) uint8."""
        ...

    def render_float(self, dt: float) -> np.ndarray:
        """Same as step() but returns (H, W, 3) float32 in [0, 1]. Used by Scope."""
        ...
```

---

## Scope Plugin Architecture

### File Structure

```
plugins/cellular_automata/
├── __init__.py
├── __main__.py          # Entry point: python -m cellular_automata coral
├── plugin.py            # Scope registration (register_pipelines)
├── pipeline.py          # CAPipeline class (Scope interface)
├── simulator.py         # CASimulator (pure logic, extracted from viewer.py)
├── viewer.py            # pygame Viewer (wraps CASimulator + handles display)
├── iridescent.py        # Color pipeline (add render_float())
├── engine_base.py       # CAEngine abstract base
├── lenia.py             # Lenia FFT engine
├── smoothlife.py        # SmoothLife engine
├── mnca.py              # MNCA engine
├── gray_scott.py        # Gray-Scott engine
├── smoothing.py         # EMA + LFO systems
├── presets.py           # All preset definitions
└── controls.py          # pygame UI widgets (local-only, never imported by pipeline)
```

### pyproject.toml Entry Point

```toml
[project]
name = "cellular-automata-scope-plugin"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "torch>=2.0.0",   # Required for Scope THWC tensor return
]

[project.entry-points."scope"]
cellular_automata = "cellular_automata.plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### plugin.py

```python
from .pipeline import CAPipeline

def register_pipelines(registry):
    registry.register(
        name="cellular_automata",
        pipeline_class=CAPipeline,
        description="Living cellular automata organism as video source"
    )
```

### pipeline.py (CAPipeline)

The pipeline is text-only (no `prepare()` method — CA generates its own frames). It holds a `CASimulator` instance and calls `render_float()` each frame.

```python
import torch
import numpy as np
from .simulator import CASimulator


class CAPipeline:
    def __init__(self, preset: str = "coral", sim_size: int = 1024):
        # Load-time params: preset, sim_size
        # These are fixed at pipeline load
        self.simulator = CASimulator(preset_key=preset, sim_size=sim_size)
        self._last_time = None

    def __call__(self, prompt: str = "", **kwargs) -> torch.Tensor:
        # Runtime params — ALL read from kwargs, never cached in __init__
        preset = kwargs.get("preset", None)
        speed = kwargs.get("speed", 1.0)
        hue = kwargs.get("hue", 0.25)
        brightness = kwargs.get("brightness", 1.0)
        thickness = kwargs.get("thickness", 0.0)
        # Flow fields
        flow_rotate = kwargs.get("flow_rotate", None)
        flow_swirl = kwargs.get("flow_swirl", None)
        flow_vortex = kwargs.get("flow_vortex", None)
        flow_bubble = kwargs.get("flow_bubble", None)

        # Apply runtime params to simulator
        runtime = {k: v for k, v in {
            "speed": speed, "hue": hue, "brightness": brightness,
            "thickness": thickness, "flow_rotate": flow_rotate,
            "flow_swirl": flow_swirl, "flow_vortex": flow_vortex,
            "flow_bubble": flow_bubble,
        }.items() if v is not None}

        if preset is not None:
            self.simulator.apply_preset(preset)

        self.simulator.set_runtime_params(**runtime)

        # Compute dt (wall-clock time between Scope calls)
        import time
        now = time.monotonic()
        if self._last_time is None:
            dt = 1.0 / 30.0  # Assume 30fps on first frame
        else:
            dt = min(now - self._last_time, 0.1)  # Cap dt
        self._last_time = now

        # Render one frame as float32 [H, W, 3] in [0, 1]
        frame_np = self.simulator.render_float(dt)  # (H, W, 3) float32

        # Convert to THWC torch tensor: (1, H, W, 3)
        tensor = torch.from_numpy(frame_np).unsqueeze(0)  # Add T dimension
        return tensor  # THWC, values in [0, 1]

    @staticmethod
    def ui_field_config():
        return {
            # Load-time (Settings panel)
            "preset": {
                "order": 1, "panel": "settings", "label": "Starting Preset",
                "choices": [
                    "coral", "amoeba", "lava_lamp", "tide_pool",
                    "sl_gliders", "sl_worms", "sl_pulse",
                    "mnca_mitosis", "mnca_worm",
                    "labyrinth", "tentacles", "medusa",
                ]
            },
            "sim_size": {
                "order": 2, "panel": "settings", "label": "Sim Resolution",
                "choices": [512, 1024]
            },
            # Runtime (Controls panel)
            "speed": {
                "order": 1, "panel": "controls", "label": "Speed",
                "min": 0.5, "max": 5.0, "step": 0.1
            },
            "hue": {
                "order": 2, "panel": "controls", "label": "Hue",
                "min": 0.0, "max": 1.0, "step": 0.01
            },
            "brightness": {
                "order": 3, "panel": "controls", "label": "Brightness",
                "min": 0.1, "max": 3.0, "step": 0.1
            },
            "thickness": {
                "order": 4, "panel": "controls", "label": "Thickness",
                "min": 0.0, "max": 20.0, "step": 0.5
            },
            "flow_rotate": {
                "order": 5, "panel": "controls", "label": "Rotation Flow",
                "min": -1.0, "max": 1.0, "step": 0.05
            },
            "flow_swirl": {
                "order": 6, "panel": "controls", "label": "Swirl Flow",
                "min": -1.0, "max": 1.0, "step": 0.05
            },
            "flow_vortex": {
                "order": 7, "panel": "controls", "label": "Vortex Flow",
                "min": -1.0, "max": 1.0, "step": 0.05
            },
        }
```

---

## Headless Operation (No pygame)

The critical design principle: `simulator.py` must never import pygame.

**pygame-free rendering path:**
- `IridescentPipeline.render()` already returns `np.ndarray (H, W, 3) uint8` — no pygame
- Add `IridescentPipeline.render_float()` that returns `(H, W, 3) float32 in [0, 1]` — divide by 255.0
- All scipy operations (gaussian_filter, zoom, map_coordinates) are import-guarded with `try/except` already — they fall back gracefully
- Engine `.step()` methods are pure numpy — no display dependency
- All flow field / containment / noise logic in viewer.py is pure numpy — moves to simulator.py cleanly

**Headless test harness:**

```python
# Verify no pygame import:
import importlib
import sys
# Block pygame
sys.modules['pygame'] = None

from cellular_automata.simulator import CASimulator  # Must not crash
sim = CASimulator(preset_key="coral")
frame = sim.render_float(0.033)  # 30fps tick
assert frame.shape == (1024, 1024, 3)
assert frame.min() >= 0.0 and frame.max() <= 1.0
```

**`__main__.py` stays pygame-dependent** (only imported when running as standalone):
```python
# __main__.py
from .viewer import Viewer   # Viewer imports pygame — that's fine here
```

**`viewer.py` imports pygame at module level** — stays isolated from `simulator.py` and `pipeline.py`.

---

## CuPy GPU Abstraction Layer

### Strategy: Conditional Import with Backend Swapping

Do NOT create parallel GPU code paths for each engine. Instead, create a thin backend module that switches `np`/`scipy` to `cupy`/`cupyx` based on availability. All engine and simulator code imports from the backend.

```
Phase 3 goal: one-line backend switch toggles all GPU acceleration.
```

### Backend Module Design

```python
# cellular_automata/backend.py

def get_backend():
    """Return (np, scipy_ndimage) — CuPy if available, numpy otherwise."""
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpnd
        # Verify GPU is actually accessible
        cp.zeros(1)
        return cp, cpnd
    except (ImportError, Exception):
        import numpy as np
        try:
            import scipy.ndimage as nd
        except ImportError:
            nd = None
        return np, nd

np, ndimage = get_backend()
```

Then in each engine file:
```python
# lenia.py
from .backend import np  # Replaces: import numpy as np

class Lenia(CAEngine):
    def step(self):
        world_fft = np.fft.rfft2(self.world)  # numpy or cupy — same API
        ...
```

And in simulator.py:
```python
from .backend import np, ndimage
# All gaussian_filter, map_coordinates, zoom calls use ndimage
```

### numpy vs CuPy API Differences

| numpy/scipy operation | CuPy equivalent | Notes |
|----------------------|-----------------|-------|
| `np.fft.rfft2()` | `cp.fft.rfft2()` | Drop-in — Lenia kernel convolution |
| `np.fft.irfft2()` | `cp.fft.irfft2()` | Drop-in |
| `scipy.ndimage.gaussian_filter()` | `cupyx.scipy.ndimage.gaussian_filter()` | Drop-in |
| `scipy.ndimage.zoom()` | `cupyx.scipy.ndimage.zoom()` | Drop-in |
| `scipy.ndimage.map_coordinates()` | `cupyx.scipy.ndimage.map_coordinates()` | Drop-in — advection |
| `np.random.randn()` | `cp.random.randn()` | Drop-in — noise pool |
| `np.zeros(..., dtype=np.float32)` | `cp.zeros(..., dtype=cp.float32)` | Drop-in |
| `np.ogrid[...]` | `cp.ogrid[...]` | Drop-in |

**Critical difference: data transfer.** When returning to Scope, convert from GPU to CPU:
```python
# In render_float() inside CASimulator:
frame_np = self.iridescent.render_float(dt)  # Returns cupy array if GPU
if hasattr(frame_np, 'get'):
    frame_np = frame_np.get()  # cupy → numpy (GPU → CPU)
return frame_np  # numpy array for torch.from_numpy()
```

**Pre-allocate on GPU.** The noise pool, containment fields, flow fields — all pre-computed at init. On GPU these live in VRAM. The 1024x1024 float32 world + buffers = ~16MB — trivial for RTX 5090's 32GB.

### Performance Targets

| Operation | CPU (numpy) | GPU (CuPy) target | Bottleneck |
|-----------|-------------|-------------------|------------|
| Lenia step (FFT) | ~80ms | ~5ms | cuFFT |
| SmoothLife step (FFT) | ~80ms | ~5ms | cuFFT |
| MNCA step | ~40ms | ~3ms | GPU parallelism |
| Gray-Scott step | ~25ms | ~2ms | Stencil ops |
| Advection (map_coordinates) | ~60ms | ~4ms | Texture interpolation |
| Bloom blur | ~30ms | ~2ms | Gaussian kernel |
| Iridescent render | ~8ms | ~1ms | LUT lookup |
| **Total** | **~120-300ms** | **~10-20ms** | |

CPU currently achieves 10-15 FPS. GPU target: 30+ FPS at 1024x1024. GPU to CPU transfer for one frame: ~4ms (12MB via PCIe).

---

## Data Flow: Scope → CA → Scope

```
Scope scheduler (each frame)
    │
    ↓  __call__(prompt="", **kwargs)
CAPipeline
    │  reads: preset, speed, hue, brightness, flow_*, thickness from kwargs
    │
    ↓  set_runtime_params(**runtime)
CASimulator
    │  updates: LFO bases, flow strengths, iridescent hue/brightness
    │
    ↓  step(dt)
    │
    ├── LFO systems: lfo.update(dt) → modulated params
    ├── engine.set_params(**modulated)
    ├── for _ in speed_accumulator steps:
    │       engine.step()                    ← numpy or CuPy
    │       _advect(engine.world)            ← map_coordinates (numpy/CuPy)
    │       engine.world *= _containment     ← element-wise mul
    │       _apply_noise_and_stir()          ← noise pool + stir field
    ├── _manage_coverage()                   ← check mass, reseed if dead
    │
    ↓  render_float(dt)
IridescentPipeline
    ├── compute_signals(world)               ← edges, velocity
    ├── compute_color_parameter(...)         ← weighted blend → t [0,1]
    ├── t_buffer += color_offset             ← spatial zones
    ├── _advance_hue_lfo(lfo_phase, dt)      ← hue cycling
    ├── LUT lookup (2D LUT, cached)          ← t×alpha → uint8 RGB
    ↓  returns (H, W, 3) float32 [0, 1]
    │
    ↓  GPU→CPU transfer if CuPy (.get())
    │
    ↓  torch.from_numpy().unsqueeze(0)      ← (H, W, 3) → (1, H, W, 3)
    │
    ↓  THWC tensor [0, 1] → Scope
Scope feeds into Krea Realtime + LoRAs
```

---

## Build Order and Dependencies

### Phase 2A: Extract CASimulator (prerequisite for everything)

Extract simulation logic from `viewer.py` into `simulator.py`. This is a refactor with no new functionality.

**Dependencies:** None — pure extraction.
**Test:** `python -m cellular_automata coral` still works identically after extraction.
**Definition of done:** `CASimulator` instantiates, `render_float()` returns correct array, viewer delegates to it.

### Phase 2B: Write Scope Plugin (depends on 2A)

Create `plugin.py` and `pipeline.py`. Wire `CAPipeline` to `CASimulator`.

**Dependencies:** Phase 2A complete, CASimulator has `render_float()`.
**Test:** Install plugin with `uv run daydream-scope install .`, verify it appears in Scope UI.
**Definition of done:** Scope UI shows "cellular_automata" pipeline, selecting it produces video frames.

### Phase 2C: Wire Runtime Params (depends on 2B)

Map all Scope kwargs through `set_runtime_params()`. Verify Scope sliders update live.

**Dependencies:** Phase 2B working.
**Test:** Move Scope speed slider → organism visibly speeds up. Move hue slider → colors shift.
**Definition of done:** All 4 runtime param categories (speed, hue/brightness, flow, preset) update without pipeline restart.

### Phase 3A: Backend Module (depends on 2A, prerequisite for 3B)

Create `backend.py` with conditional numpy/CuPy import. Update all engine files to `from .backend import np`.

**Dependencies:** Phase 2A (CASimulator extracted, clear import structure).
**Test:** On CPU machine, `backend.py` returns numpy. On GPU machine (RunPod), returns CuPy.
**Definition of done:** All engines and simulator work unchanged with numpy backend.

### Phase 3B: Port Engines to Backend (depends on 3A)

Verify each engine's `step()` works with CuPy. Fix any non-portable operations.

**Dependencies:** Phase 3A.
**Critical items to check:**
- `np.fft.rfft2` in Lenia — CuPy is identical
- `map_coordinates` in advection — cupyx.scipy.ndimage is identical
- `np.random.randn` in noise pool — CuPy is identical
- `np.ogrid` in field builders — CuPy is identical
- Any calls to `scipy.ndimage.maximum_filter` in `_dilate_world` — use `cupyx.scipy.ndimage.maximum_filter`

**Test:** Enable CuPy backend on RunPod, run 100 frames, verify same visual output as CPU.

### Phase 4: RunPod Deployment (depends on Phases 2 + 3)

**Dependencies:** Scope plugin installed (Phase 2), CuPy backend working (Phase 3).
**Steps:**
1. Build container with CuPy + scipy + the CA plugin installed
2. Start RunPod pod (RTX 5090)
3. `daydream-scope install cellular_automata`
4. Select CA pipeline in Scope, verify frames flow to Krea Realtime
5. Performance profile: confirm 30+ FPS at 1024x1024

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Making plugin.py Import pygame

**What people do:** Import from viewer.py in plugin.py, dragging pygame in.
**Why wrong:** Headless Scope containers don't have a display. `pygame.init()` crashes on import in headless environments.
**Instead:** `plugin.py` imports only from `pipeline.py`. `pipeline.py` imports only from `simulator.py`. `viewer.py` is never imported by the plugin chain.

### Anti-Pattern 2: Storing Runtime Params in __init__

**What people do:**
```python
# WRONG
class CAPipeline:
    def __init__(self, speed=1.0, hue=0.25, ...):
        self.speed = speed   # Stored in init
    def __call__(self, **kwargs):
        # Uses self.speed from init — never updates
```
**Why wrong:** Scope UI sliders call `__call__(**kwargs)` each frame. If params are stored in `__init__`, Scope can only change them by reloading the pipeline.
**Instead:** Always read from `kwargs` in `__call__`. Document this pattern explicitly.

### Anti-Pattern 3: Parallelizing numpy + CuPy Code Paths

**What people do:** Duplicate every engine method with `if use_gpu: cupy_version() else: numpy_version()`.
**Why wrong:** Doubles maintenance burden. Any algorithm change requires two edits. Divergence is inevitable.
**Instead:** Single backend abstraction (`backend.py`). The API difference between numpy and CuPy is near-zero for all operations used here.

### Anti-Pattern 4: Full GPU→CPU Transfer Every Frame

**What people do:** `frame_cpu = cupy_array.get()` called multiple times per frame (once for iridescent pipeline, once for advection, once for bloom...).
**Why wrong:** PCIe transfer costs ~4ms per 12MB. Multiple transfers per frame: ~20ms overhead, killing GPU gains.
**Instead:** Keep all intermediate buffers on GPU. Transfer to CPU exactly once per frame at the end of `render_float()`.

### Anti-Pattern 5: Hard-Coding sim_size=1024

**What people do:** Hardcode resolution in pipeline for simplicity.
**Why wrong:** GS runs at 512 for performance. The user should be able to choose. The `sim_size` load-time param already handles this.
**Instead:** Expose `sim_size` as a load-time param with choices [512, 1024]. CASimulator's `_apply_preset()` already handles per-engine size selection (`gray_scott` → 512, others → 1024).

### Anti-Pattern 6: torch.from_numpy on cupy Arrays

**What people do:**
```python
tensor = torch.from_numpy(cupy_array)  # CRASHES — cupy is not numpy
```
**Why wrong:** `torch.from_numpy()` only accepts numpy arrays.
**Instead:** Always `.get()` first if on GPU:
```python
arr = frame  # cupy array
if hasattr(arr, 'get'):
    arr = arr.get()  # → numpy
tensor = torch.from_numpy(arr)
```

---

## Integration Points

### Scope ↔ CA Pipeline

| Scope expects | CA provides | Where |
|---------------|-------------|-------|
| `register_pipelines(registry)` hook | `plugin.py::register_pipelines()` | Entry point |
| `pipeline_class` with `__call__(**kwargs)` | `CAPipeline.__call__()` | `pipeline.py` |
| `__call__` returns THWC tensor | `torch.Tensor (1, H, W, 3) in [0,1]` | `pipeline.py` |
| `ui_field_config()` classmethod | `CAPipeline.ui_field_config()` | `pipeline.py` |
| pyproject.toml entry point | `[project.entry-points."scope"]` | `pyproject.toml` |
| No `prepare()` for text-only | CAPipeline has no `prepare()` | By omission |

### Existing CA Code ↔ New Simulator

| CASimulator calls | Lives in | Status |
|-------------------|----------|--------|
| `Lenia()`, `SmoothLife()`, `MNCA()`, `GrayScott()` | engine files | Ready |
| `IridescentPipeline()` | `iridescent.py` | Ready — add `render_float()` |
| `LeniaLFOSystem()` etc. | `smoothing.py` | Ready — move from viewer.py |
| `SmoothedParameter()` | `smoothing.py` | Ready |
| `get_preset()` | `presets.py` | Ready |
| `gaussian_filter`, `map_coordinates`, `zoom` | scipy/CuPy | Already import-guarded |
| `np.fft.rfft2` | Lenia.step() | Will port to backend.np |

### Viewer ↔ CASimulator (post-refactor)

```python
class Viewer:
    def __init__(self, preset="coral"):
        self.sim = CASimulator(preset_key=preset)

    def run(self):
        pygame.init()
        # ... setup ...
        while self.running:
            # ... event handling ...
            if not self.paused:
                dt = clock.get_time() / 1000.0
                self.sim.set_runtime_params(**self._collect_slider_values())
                rgb_u8 = self.sim.step(dt)  # (H, W, 3) uint8
                surface = pygame.surfarray.make_surface(rgb_u8.swapaxes(0, 1))
                # ... display ...
```

---

## Sources

- `plugins/example_plugin/pipeline.py` — Scope pipeline interface (THWC, kwargs, ui_field_config)
- `plugins/example_plugin/pyproject.toml` — Entry point format verified
- `plugins/cellular_automata/viewer.py` — Extraction boundary analyzed
- `plugins/cellular_automata/iridescent.py` — Confirmed pygame-free, returns uint8
- `plugins/cellular_automata/engine_base.py` — Confirmed pure numpy
- CuPy documentation: CuPy provides near-identical API to numpy; cupyx.scipy.ndimage mirrors scipy.ndimage for all functions used (gaussian_filter, zoom, map_coordinates)
- Scope plugin contract: text-only pipeline omits `prepare()`, returns THWC float32 tensor

---
*Architecture research for: Scope plugin integration + CuPy GPU port*
*Researched: 2026-02-17*
