# Phase 4: Scope Plugin Wrapper - Research

**Researched:** 2026-02-18
**Domain:** DayDream Scope plugin packaging, Python entry points, Pydantic pipeline config, THWC tensor output
**Confidence:** HIGH — official Scope docs verified, local codebase audited, prior architecture research confirmed

---

## Summary

Phase 3 is complete: `CASimulator` is extracted into `simulator.py` with zero pygame dependency, and `viewer.py` is a thin 474-line display wrapper. The path to a working Scope plugin is well-understood and mostly risk-free. The primary work is writing three new files: `pyproject.toml`, `plugin.py`, and `pipeline.py`, then installing and testing the plugin.

Two competing API patterns exist in this codebase. The **local example_plugin** uses a simpler pattern (no `@hookimpl`, no `BasePipelineConfig`, `registry.register(name=, pipeline_class=, description=)`). The **official Scope docs** define a formal pattern (`@hookimpl`, `BasePipelineConfig` with Pydantic fields, `Pipeline` ABC, `get_config_class()`, `register(PipelineClass)`). The requirements in this project (PLUG-01 through PLUG-10) explicitly name the formal API. Both API styles ultimately install via the same `uv run daydream-scope install .` command and the same `[project.entry-points."scope"]` entry point.

The largest single risk is the **API mismatch**: if the installed Scope version uses the simpler registry pattern (as in the local example), using `@hookimpl` and `BasePipelineConfig` will cause import errors. The recommended approach is to **start with the simpler example_plugin pattern** (which is confirmed working from local code) and add `BasePipelineConfig` only if the Scope UI requires it for slider configuration. All requirements can be satisfied with either pattern.

**Primary recommendation:** Follow the local example_plugin pattern for `plugin.py` and `pipeline.py` (it's verified working). Use `pyproject.toml` hatchling build with `[project.entry-points."scope"]`. The `CASimulator.render_float()` → `torch.from_numpy().unsqueeze(0)` conversion is the core output path. Defer `BasePipelineConfig`/`@hookimpl` unless Scope rejects the simpler pattern.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PLUG-01 | Plugin Registration — `plugin.py` with `@hookimpl` decorated `register_pipelines(register)` | Official docs confirm this pattern. Local example_plugin uses simpler pattern without decorator. Start with simple pattern; add `@hookimpl` if Scope requires it. |
| PLUG-02 | Pipeline Config Schema — Pydantic `BasePipelineConfig` with `pipeline_id`, `pipeline_name`, etc. | Official docs confirm full Pydantic schema. Local example uses `ui_field_config()` static dict instead. Both expose sliders; `BasePipelineConfig` is more formal but requires `scope.core.pipelines.base_schema` import. |
| PLUG-03 | Pipeline Class — `CAPipeline` inheriting `scope.core.pipelines.interface.Pipeline`, `get_config_class()`, `__call__()` | Official docs confirm ABC. Local example uses plain class. `CASimulator.render_float()` is the rendering call; `torch.from_numpy().unsqueeze(0)` gives THWC tensor. |
| PLUG-04 | Runtime kwargs — ALL user-controllable params read from `kwargs` in `__call__()` every frame | Confirmed pattern in both local example and official docs. `CASimulator.set_runtime_params(**kwargs)` is the delegation call. Preset changes go through `simulator.apply_preset(key)`. |
| PLUG-05 | THWC Tensor Output — `{"video": tensor}` where tensor is `torch.Tensor (1, H, W, 3) float32 [0,1]` | Official docs confirm dict return `{"video": tensor}`. Local example returns raw tensor. The `render_float()` returns `(H,W,3) float32 [0,1]`; `.copy()` + `torch.from_numpy()` + `.unsqueeze(0)` gives THWC. |
| PLUG-06 | Wall-Clock dt — `time.perf_counter()` between `__call__()` calls, clamped to `[0.001, 0.1]` | CASimulator already accepts `dt` in `render_float(dt)`. Pipeline tracks `_last_time` and computes `dt = perf_counter() - _last_time`. LFO breathing accuracy depends on this. |
| PLUG-07 | pyproject.toml — hatchling, `[project.entry-points."scope"]`, no pygame, exclude viewer/controls/__main__ | Confirmed from example_plugin pyproject.toml. `[tool.hatch.build.targets.wheel]` exclude list prevents pygame-dependent files from installing. |
| PLUG-08 | Warmup Strategy — Engine warmup deferred out of `__init__()` | `_apply_preset()` in CASimulator currently runs 200-1000 warmup steps. Need a `_warmed_up` flag, run warmup on first `__call__()` or background thread. |
| PLUG-09 | Install and Test — `uv run daydream-scope install .` succeeds | Confirmed from example_plugin README. Entry point package name must match installable package path exactly. |
| PLUG-10 | No pygame in Import Chain — verified with sys.modules guard | `simulator.py` is already pygame-free (Phase 3 complete). `plugin.py` → `pipeline.py` → `simulator.py` chain never touches viewer.py or controls.py. |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hatchling | any (resolved by uv) | Build backend for pyproject.toml | What the local example_plugin uses; uv resolves it automatically |
| torch | >=2.0.0 | THWC tensor output | Scope expects `torch.Tensor` not numpy arrays |
| numpy | >=1.24.0 | Simulation arrays | Already used throughout CA engines |
| scipy | >=1.10.0 | gaussian_filter, zoom, map_coordinates | Already guarded with try/except in simulator.py; needed for advection and dilation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| time (stdlib) | — | Wall-clock dt tracking | `time.perf_counter()` between `__call__()` invocations |
| threading (stdlib) | — | Background warmup | If warmup delay is unacceptable on first call |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| hatchling | setuptools | hatchling is what the example uses; setuptools is more familiar but unnecessary complexity |
| `time.perf_counter()` | `time.monotonic()` | Both work; `perf_counter()` has higher resolution on some platforms; prior architecture doc uses `monotonic()` — either is fine |
| Background thread warmup | First-call warmup | Background thread is cleaner UX but more complex; first-call is simpler and likely fine since GS warmup on CPU is the slow case |

**Installation:**
```bash
# Inside the cellular_automata package directory:
uv run daydream-scope install .

# Or for local editable install (development):
uv run daydream-scope install -e /path/to/plugins/cellular_automata
```

---

## Architecture Patterns

### Recommended Project Structure

```
plugins/cellular_automata/         <- package root (this is what gets installed)
├── pyproject.toml                 <- MUST exist; hatchling build + entry point
├── __init__.py                    <- "Cellular Automata - Lenia and beyond" (exists)
├── __main__.py                    <- pygame runner; EXCLUDED from wheel
├── plugin.py                      <- NEW: register_pipelines() hook
├── pipeline.py                    <- NEW: CAPipeline class
├── simulator.py                   <- DONE: CASimulator (Phase 3 complete)
├── viewer.py                      <- pygame wrapper; EXCLUDED from wheel
├── controls.py                    <- pygame UI; EXCLUDED from wheel
├── iridescent.py                  <- color pipeline (no pygame)
├── engine_base.py                 <- abstract engine base (no pygame)
├── lenia.py / smoothlife.py / ... <- engines (no pygame)
├── presets.py                     <- preset definitions (no pygame)
└── smoothing.py                   <- EMA + LFO systems (no pygame)
```

### Pattern 1: Plugin Registration (Simpler — Confirmed Working)

The local `example_plugin` pattern. Use this first; it's verified working from the codebase.

```python
# plugin.py
from .pipeline import CAPipeline

def register_pipelines(registry):
    """Called when Scope loads the plugin."""
    registry.register(
        name="cellular_automata",
        pipeline_class=CAPipeline,
        description="Living cellular automata organism as video source"
    )
```

**Source:** `/Users/agi/Code/daydream_scope/plugins/example_plugin/plugin.py` (HIGH confidence — in repo)

### Pattern 2: Plugin Registration (Formal — Official API)

The official docs pattern using `@hookimpl`. Use this if the simpler pattern fails or if `scope.core.plugins.hookspecs` is importable.

```python
# plugin.py
from scope.core.plugins.hookspecs import hookimpl
from .pipeline import CAPipeline

@hookimpl
def register_pipelines(register):
    register(CAPipeline)
```

**Source:** Official Scope docs at `docs.daydream.live/scope/guides/plugin-development` (MEDIUM confidence — docs page verified but could be version-dependent)

**KEY DIFFERENCE from local example:**
- Official: `register(PipelineClass)` — positional arg, class only
- Local example: `registry.register(name=, pipeline_class=, description=)` — keyword args, registry object

### Pattern 3: Pipeline Class (Simple — Confirmed Working)

Matches the local example_plugin exactly. Text-only pipeline, no `prepare()`.

```python
# pipeline.py
import time
import torch
import numpy as np
from .simulator import CASimulator

class CAPipeline:
    """Text-only CA pipeline — generates CA video frames as Scope input source."""

    def __init__(self, sim_size: int = 512, preset: str = "coral"):
        """Load-time initialization. Creates CASimulator, defers warmup."""
        # Load-time params (set once at pipeline load)
        self._sim_size = sim_size
        self._default_preset = preset
        # Simulator is created now; warmup deferred to first __call__()
        self.simulator = CASimulator(preset_key=preset, sim_size=sim_size)
        self._warmed_up = False
        # Wall-clock time tracking for LFO accuracy
        self._last_time = None

    def __call__(self, prompt: str = "", **kwargs) -> dict:
        """Generate one CA frame. All user-controllable params from kwargs."""

        # Deferred warmup (runs once on first call, not at init)
        if not self._warmed_up:
            self._run_warmup()
            self._warmed_up = True

        # Read ALL runtime params from kwargs every frame
        preset = kwargs.get("preset", None)
        speed = kwargs.get("speed", 1.0)
        hue = kwargs.get("hue", 0.25)
        brightness = kwargs.get("brightness", 1.0)
        thickness = kwargs.get("thickness", 0.0)
        reseed = kwargs.get("reseed", False)

        # Apply preset change if requested
        if preset is not None and preset != self.simulator.preset_key:
            self.simulator.apply_preset(preset)
            self._warmed_up = False  # Retrigger warmup for new engine

        # Apply all runtime params to simulator
        self.simulator.set_runtime_params(
            speed=speed, hue=hue, brightness=brightness,
            thickness=thickness, reseed=reseed
        )

        # Wall-clock dt for LFO accuracy (NOT fixed dt)
        now = time.perf_counter()
        if self._last_time is None:
            dt = 1.0 / 30.0  # Assume 30fps on first frame
        else:
            dt = now - self._last_time
        dt = max(0.001, min(dt, 0.1))  # Clamp to [0.001, 0.1]
        self._last_time = now

        # Render one frame as float32 [0,1]
        frame_np = self.simulator.render_float(dt)  # (H, W, 3) float32

        # Convert to THWC torch tensor: (1, H, W, 3)
        tensor = torch.from_numpy(frame_np.copy()).unsqueeze(0)

        return {"video": tensor}  # Scope expects dict with "video" key

    def _run_warmup(self):
        """Run warmup steps so first frame shows developed structure."""
        engine_name = self.simulator.engine_name
        # Already run 200-1000 steps in _apply_preset() inside CASimulator.__init__
        # This is a placeholder in case we need additional warmup here
        # The real warmup happens in CASimulator._apply_preset() -> 200 steps for Lenia, 1000 for GS
        pass

    @staticmethod
    def ui_field_config():
        """Configure how parameters appear in Scope UI."""
        return {
            # Load-time (Settings panel — requires pipeline reload to change)
            "sim_size": {
                "order": 1, "panel": "settings", "label": "Sim Resolution",
                "choices": [512, 1024], "is_load_param": True,
            },
            # Runtime (Controls panel — updates live per-frame)
            "preset": {
                "order": 1, "panel": "controls", "label": "Preset",
                "choices": [
                    "coral", "amoeba", "jellyfish", "lava_lamp", "tide_pool",
                    "heartbeat", "nebula",
                    "sl_gliders", "sl_worms", "sl_pulse",
                    "mnca_mitosis", "mnca_worm",
                    "labyrinth", "tentacles", "medusa",
                ],
            },
            "speed": {
                "order": 2, "panel": "controls", "label": "Speed",
                "min": 0.5, "max": 5.0, "step": 0.1,
            },
            "hue": {
                "order": 3, "panel": "controls", "label": "Hue",
                "min": 0.0, "max": 1.0, "step": 0.01,
            },
            "brightness": {
                "order": 4, "panel": "controls", "label": "Brightness",
                "min": 0.1, "max": 3.0, "step": 0.1,
            },
            "thickness": {
                "order": 5, "panel": "controls", "label": "Thickness",
                "min": 0.0, "max": 20.0, "step": 0.5,
            },
            "reseed": {
                "order": 6, "panel": "controls", "label": "Reseed",
                "type": "toggle",
            },
        }
```

### Pattern 4: Formal Pipeline Class (Official API with BasePipelineConfig)

Use only if the simpler pattern fails and Scope requires `BasePipelineConfig`.

```python
# pipeline.py (formal version)
from pydantic import Field
from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from scope.core.pipelines.interface import Pipeline
from .simulator import CASimulator

class CAPipelineConfig(BasePipelineConfig):
    pipeline_id = "cellular-automata"
    pipeline_name = "Cellular Automata"
    pipeline_description = "Bioluminescent CA organism as video source"
    supports_prompts = False
    modes = {"text": ModeDefaults(default=True)}

    # Load-time
    sim_size: int = Field(
        default=512, description="Simulation grid resolution",
        json_schema_extra=ui_field_config(order=1, label="Sim Resolution", is_load_param=True),
    )
    # Runtime
    preset: str = Field(
        default="coral", description="CA preset to run",
        json_schema_extra=ui_field_config(order=1, label="Preset"),
    )
    speed: float = Field(
        default=1.0, ge=0.5, le=5.0,
        json_schema_extra=ui_field_config(order=2, label="Speed"),
    )
    hue: float = Field(
        default=0.25, ge=0.0, le=1.0,
        json_schema_extra=ui_field_config(order=3, label="Hue"),
    )
    brightness: float = Field(
        default=1.0, ge=0.1, le=3.0,
        json_schema_extra=ui_field_config(order=4, label="Brightness"),
    )
    thickness: float = Field(
        default=0.0, ge=0.0, le=20.0,
        json_schema_extra=ui_field_config(order=5, label="Thickness"),
    )
    reseed: bool = Field(
        default=False,
        json_schema_extra=ui_field_config(order=6, label="Reseed"),
    )

class CAPipeline(Pipeline):
    @classmethod
    def get_config_class(cls):
        return CAPipelineConfig

    def __init__(self, sim_size: int = 512, preset: str = "coral", **kwargs):
        self.simulator = CASimulator(preset_key=preset, sim_size=sim_size)
        self._last_time = None

    def __call__(self, **kwargs) -> dict:
        # Same runtime pattern as simple version above
        ...
```

**Source:** Official Scope docs (MEDIUM confidence — requires `scope.core.*` imports which may fail if Scope isn't installed)

### Pattern 5: pyproject.toml

```toml
[project]
name = "cellular-automata-scope-plugin"
version = "0.2.0"
description = "Bioluminescent cellular automata video source for DayDream Scope"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "torch>=2.0.0",
]
# DO NOT add: pygame, pygame-ce

[project.entry-points."scope"]
cellular_automata = "cellular_automata.plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["cellular_automata"]
exclude = [
    "cellular_automata/viewer.py",
    "cellular_automata/controls.py",
    "cellular_automata/__main__.py",
]
```

**Source:** `/Users/agi/Code/daydream_scope/plugins/example_plugin/pyproject.toml` (HIGH confidence — in repo)

### Anti-Patterns to Avoid

- **Importing pygame anywhere in plugin.py → pipeline.py → simulator.py chain.** pygame crashes on headless RunPod (no SDL display). `simulator.py` is already clean; keep it that way.
- **Reading runtime params from `__init__()` storage instead of `kwargs`.** Scope UI sliders call `__call__(**kwargs)` each frame. Params stored at init never update.
- **Missing `.copy()` before `torch.from_numpy()`.** `render_float()` returns the internal `display_buffer` (reused pre-allocated array). Without `.copy()`, the torch tensor will point to memory that gets overwritten on the next frame.
- **Returning raw tensor instead of `{"video": tensor}` dict.** Official docs and the requirements both specify a dict return. Local example returns raw tensor — this may be an older API. Use the dict pattern.
- **Warmup blocking `__init__()`.** `_apply_preset()` currently runs 200-1000 steps at init. Scope may time out. The `CASimulator.__init__()` calls `_apply_preset()` which runs this warmup. Defer by not running warmup in `CAPipeline.__init__()` — but note the warmup already happens inside `CASimulator.__init__()`. The real fix is to refactor `CASimulator` to optionally skip warmup, or accept the first-call latency.
- **Fixed dt (e.g., 1/30.0) for LFO.** LFO periods are tuned for real time. Fixed dt breaks breathing at non-30fps rates.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Float array to THWC tensor | Custom array manipulation | `torch.from_numpy(arr.copy()).unsqueeze(0)` | One line; handles dtype and shape correctly |
| Wall-clock time | Custom timing | `time.perf_counter()` | Highest-resolution monotonic clock on Python |
| Package build | Custom setup.py | hatchling + pyproject.toml | What Scope's ecosystem already uses |
| Preset list in UI | Dynamic introspection | Static list in `ui_field_config()` | UNIFIED_ORDER already defines the ordered list |

**Key insight:** The entire CA simulation (engines, LFOs, flow fields, color) is already implemented in `CASimulator`. Phase 4 is glue code: write three files (~150 lines total), install, test.

---

## Common Pitfalls

### Pitfall 1: API Mismatch Between Example Plugin and Official Docs

**What goes wrong:** Using official API (`@hookimpl`, `scope.core.pipelines.interface.Pipeline`) but the installed Scope version uses the simpler `registry.register(name=, pipeline_class=, description=)` API — causing import errors.

**Why it happens:** The local `example_plugin` was presumably generated for the installed Scope version. The official docs may reflect a newer API. Two different API versions exist.

**How to avoid:** Start with the simpler local example pattern. If Scope rejects it, try the `@hookimpl` formal pattern. Check if `from scope.core.plugins.hookspecs import hookimpl` succeeds before using it.

**Warning signs:** Plugin doesn't appear in Scope UI; import errors in Scope logs; `from scope.core.plugins.hookspecs import hookimpl` raises `ModuleNotFoundError`.

### Pitfall 2: `render_float()` Returns Shared Buffer — Missing `.copy()`

**What goes wrong:** `IridescentPipeline.render()` writes into `self.display_buffer` (pre-allocated uint8 array). `render_float()` calls `step()` which calls `_render_frame()` which returns `rgb.copy()` — BUT if this copy is not made, the torch tensor will be backed by memory that gets overwritten on the next frame.

**Current state:** `simulator.py` line 1269: `return rgb.copy()` — the copy IS made in `_render_frame()`. Then `render_float()` does `rgb_uint8.astype(np.float32) / 255.0` which creates a new array. The pipeline's `torch.from_numpy(frame_np.copy())` adds an extra safety copy.

**How to avoid:** Always call `.copy()` before `torch.from_numpy()` in the pipeline. Even if the source is already a fresh array, the explicit copy is cheap insurance.

### Pitfall 3: Warmup Already Happens in CASimulator.__init__()

**What goes wrong:** PLUG-08 says "warmup deferred out of `__init__()`" — but `CASimulator.__init__()` calls `_apply_preset()` which runs 200-1000 warmup steps. The plugin's `__init__()` creates a `CASimulator` and therefore already pays the warmup cost during plugin load.

**What this means:** The warmup issue is inside `CASimulator._apply_preset()`, not `CAPipeline.__init__()`. Two options:
1. Add a `warmup=True` parameter to `CASimulator.__init__()` to skip warmup at creation, run it on first `render_float()` call
2. Accept the warmup at plugin load time (Scope may handle slow plugin init gracefully)

**How to avoid:** Add `warmup` flag to `CASimulator.__init__()`. Default to `warmup=True` so existing `Viewer` keeps behavior; `CAPipeline` passes `warmup=False` and runs warmup on first `render_float()` call.

### Pitfall 4: Package Name Collision with `cellular_automata`

**What goes wrong:** The entry point `cellular_automata = "cellular_automata.plugin"` means Python must be able to `import cellular_automata`. If the package at `plugins/cellular_automata/` is installed, Python expects `cellular_automata.plugin` to resolve correctly.

**How to avoid:** Verify with `python -c "from cellular_automata.plugin import register_pipelines; print('OK')"` after install. If using `-e` (editable) mode, the source directory must be in Python path.

### Pitfall 5: Preset Key Mismatch in `__call__()`

**What goes wrong:** The pipeline receives `kwargs["preset"] = "Lenia Branch"` (display name) but `CASimulator.apply_preset()` expects `kwargs["preset"] = "coral"` (internal key). The `UNIFIED_ORDER` uses internal keys; the UI may send display names or internal keys depending on how Scope maps `ui_field_config` choices.

**How to avoid:** The `choices` list in `ui_field_config()` should contain internal keys (e.g., `"coral"`, not `"Lenia Branch"`). The UI will display the key values as-is. To show display names, either use the keys as display-friendly strings or build a name→key mapping.

**Warning signs:** `get_preset(key)` returns `None` → `_apply_preset()` silently does nothing → preset doesn't change.

---

## Code Examples

Verified patterns from official sources:

### THWC Tensor Conversion (Core Output Path)

```python
# Source: simulator.py render_float() + pipeline.py pattern
# render_float() returns (H, W, 3) float32 [0, 1]
frame_np = self.simulator.render_float(dt)   # (H, W, 3) float32

# Copy: important if underlying buffer could be reused
# from_numpy: zero-copy view of numpy memory
# unsqueeze(0): adds T=1 dimension for THWC
tensor = torch.from_numpy(frame_np.copy()).unsqueeze(0)  # (1, H, W, 3) float32

return {"video": tensor}  # Dict with "video" key per official Scope API
```

### Wall-Clock dt Tracking

```python
# Source: Prior architecture research + simulator.py pattern
import time

class CAPipeline:
    def __init__(self, ...):
        self._last_time = None

    def __call__(self, **kwargs):
        now = time.perf_counter()
        if self._last_time is None:
            dt = 1.0 / 30.0        # First frame: assume 30fps
        else:
            dt = now - self._last_time
        dt = max(0.001, min(dt, 0.1))  # Clamp: no jumps, no micro-ticks
        self._last_time = now
        # dt is now wall-clock seconds since last __call__()
```

### Preset Change Detection

```python
# In __call__():
preset = kwargs.get("preset", None)
if preset is not None and preset != self.simulator.preset_key:
    self.simulator.apply_preset(preset)  # engine swap if needed
    # Note: apply_preset() runs warmup steps internally — this is acceptable
    # on preset switch (user just selected a new preset)
```

### Deferred Warmup Pattern (if needed)

```python
# Requires adding warmup= parameter to CASimulator.__init__()
class CASimulator:
    def __init__(self, preset_key="coral", sim_size=1024, warmup=True):
        ...
        self._apply_preset(self.preset_key, run_warmup=warmup)

    def _apply_preset(self, key, run_warmup=True):
        ...
        if run_warmup:
            if new_engine_name == "lenia":
                for _ in range(200): self.engine.step()
            elif new_engine_name == "gray_scott":
                for _ in range(1000): self.engine.step()

# Pipeline:
class CAPipeline:
    def __init__(self, ...):
        self.simulator = CASimulator(preset_key=preset, sim_size=sim_size, warmup=False)
        self._warmed_up = False

    def __call__(self, **kwargs):
        if not self._warmed_up:
            # Run warmup on first call (not at plugin load)
            engine = self.simulator.engine_name
            steps = 1000 if engine == "gray_scott" else 200
            for _ in range(steps):
                self.simulator.engine.step()
            self._warmed_up = True
        ...
```

### pyproject.toml Exclude Pattern

```toml
# Prevent pygame-dependent files from installing into Scope
[tool.hatch.build.targets.wheel]
packages = ["cellular_automata"]
exclude = [
    "cellular_automata/viewer.py",
    "cellular_automata/controls.py",
    "cellular_automata/__main__.py",
]
```

### Headless Import Verification Test

```python
# Verify no pygame in plugin import chain:
import sys
sys.modules["pygame"] = None   # Block pygame
from cellular_automata.plugin import register_pipelines  # Must not crash
from cellular_automata.pipeline import CAPipeline        # Must not crash
sim = CAPipeline()
frame_dict = sim()
assert "video" in frame_dict
tensor = frame_dict["video"]
assert tensor.shape == (1, 512, 512, 3)   # (T, H, W, C) THWC
assert tensor.dtype == torch.float32
assert tensor.min() >= 0.0 and tensor.max() <= 1.0
print("All assertions passed")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| All logic in viewer.py with pygame | CASimulator extracted, pygame-free | Phase 3 (2026-02-18) | Plugin chain can import without pygame |
| Manual numpy→pygame Surface | numpy array returned from `_render_frame()` | Phase 3 | Direct numpy→torch conversion now possible |
| 1765-line viewer.py | 474-line thin wrapper + simulator.py | Phase 3 | Clear separation of concerns |
| No pyproject.toml in CA package | Need to create for Phase 4 | Phase 4 task | Plugin system requires it |

**Deprecated/outdated:**
- `viewer.py` heavy simulation logic: Removed in Phase 3; simulator.py is the authoritative sim path
- `_render_frame()` returning pygame.Surface: Now returns `np.ndarray (H,W,3) uint8`
- CCA/Life/Excitable engines: Not in `ENGINE_CLASSES` in simulator.py; excluded from headless path

---

## Open Questions

1. **Which Scope API version is installed?**
   - What we know: Local `example_plugin` uses simpler `registry.register(name=, pipeline_class=, description=)` without `@hookimpl`. Official docs show `@hookimpl` + `register(PipelineClass)`.
   - What's unclear: Which version the current Scope installation expects. The return format (`{"video": tensor}` vs raw tensor) is also in question.
   - Recommendation: Try the simple pattern first (matches local example). If Scope UI doesn't show the pipeline, try the formal `@hookimpl` pattern. Test the return format with simple tensor first if the dict causes errors.

2. **Does warmup in `CASimulator.__init__()` cause plugin load timeout?**
   - What we know: GS warmup is 1000 steps (slow on CPU). Lenia is 200 steps. The current `CASimulator.__init__()` calls `_apply_preset()` which runs warmup.
   - What's unclear: Whether Scope has an init timeout, and how long warmup actually takes (may be acceptable).
   - Recommendation: Profile warmup time first (`time.time()` around `CASimulator("coral", 512)`). If under 5 seconds, it's probably fine. If over 5 seconds, add the `warmup=False` parameter.

3. **Do Scope UI `choices` in `ui_field_config()` accept internal preset keys or require display names?**
   - What we know: `presets.py` uses keys like `"coral"`, `"amoeba"`. The `PRESETS` dict maps key→`{"name": "Lenia Branch", ...}`. The viewer shows display names in the button row.
   - What's unclear: Whether Scope sends the choice value as-is (key string) or requires name mapping.
   - Recommendation: Start with internal keys in choices list. If Scope sends display names, add a name→key mapping in `__call__()`.

4. **Does `__call__()` receive a `prompt` positional arg or only kwargs?**
   - What we know: Local example shows `def __call__(self, prompt: str, **kwargs)`. Official docs show `def __call__(self, **kwargs)`.
   - What's unclear: Whether Scope always provides a `prompt` argument for text-only pipelines.
   - Recommendation: Define `def __call__(self, prompt: str = "", **kwargs)` to handle both.

---

## Sources

### Primary (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/plugin.py` — registry.register() pattern without @hookimpl
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/pipeline.py` — THWC tensor output, ui_field_config structure, kwargs pattern, ui_field_config() static dict
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/pyproject.toml` — entry point format, hatchling build, dependencies
- `/Users/agi/Code/daydream_scope/plugins/cellular_automata/simulator.py` — CASimulator API: `render_float(dt)`, `apply_preset(key)`, `set_runtime_params(**kwargs)`, confirmed pygame-free
- `/Users/agi/Code/daydream_scope/.planning/phases/03-extract-casimulator/03-02-SUMMARY.md` — Phase 3 confirmed complete; viewer.py thin wrapper; render_float() returns (H,W,3) float32 [0,1]
- `/Users/agi/Code/daydream_scope/.planning/STATE.md` — Confirmed decisions: BasePipelineConfig+Pydantic, @hookimpl+register(), {"video": tensor} dict return, enum/dropdown for preset selector

### Secondary (MEDIUM confidence)
- `https://docs.daydream.live/scope/guides/plugin-development` — Official docs for @hookimpl, BasePipelineConfig, ModeDefaults, Pipeline ABC, get_config_class(), ui_field_config with json_schema_extra, {"video": tensor} dict return. Verified from live page 2026-02-18.
- `/Users/agi/Code/daydream_scope/.planning/research/ARCHITECTURE.md` — Prior architecture research with full CAPipeline code sketch, data flow diagram, component boundaries (researched 2026-02-17)
- `/Users/agi/Code/daydream_scope/.planning/research/STACK.md` — Stack details for plugin packaging, CuPy, headless (researched 2026-02-17)
- `/Users/agi/Code/daydream_scope/.planning/research/PITFALLS.md` — 17 detailed pitfalls covering pygame import, kwargs pattern, THWC shape, warmup, dt tracking (researched 2026-02-17)

### Tertiary (LOW confidence)
- None — no unverified claims present in research.

---

## Metadata

**Confidence breakdown:**
- Standard stack (hatchling, torch, pyproject.toml format): HIGH — verified from local example_plugin
- Architecture (CAPipeline pattern, render_float call, THWC conversion): HIGH — verified from simulator.py + official docs + prior research
- Pitfalls (pygame import, kwargs pattern, warmup, dt): HIGH — derived from direct code audit + prior pitfall research
- API version discrepancy (hookimpl vs simple registry, dict vs raw tensor): MEDIUM — both patterns found; correct one unknown until tested

**Research date:** 2026-02-18
**Valid until:** 2026-03-18 (stable API; verify before re-using if Scope version changes)

---

## Summary: What the Planner Needs to Know

Phase 4 has three outputs — `pyproject.toml`, `plugin.py`, `pipeline.py` — plus one optional `CASimulator` change (warmup flag). The core simulation is done. The risk is the API version ambiguity; mitigate by starting with the simple pattern and testing before adding `BasePipelineConfig`.

**Build order for tasks:**
1. Create `pyproject.toml` with hatchling + entry point (5 min, no code risk)
2. Create `plugin.py` with `register_pipelines()` — simple pattern first (10 min)
3. Create `pipeline.py` with `CAPipeline` — `CASimulator.render_float(dt)` + THWC conversion (20 min)
4. Optionally add `warmup=False` to `CASimulator.__init__()` if warmup timing is a problem (10 min)
5. Install with `uv run daydream-scope install .` and verify in Scope UI (varies)
6. Verify all 10 success criteria (PLUG-01 through PLUG-10) including the no-pygame assertion

---
*Phase: 04-scope-plugin-wrapper*
*Researched: 2026-02-18*
*Domain: Scope plugin packaging, Python entry points, Pydantic pipeline config, THWC tensor output*
