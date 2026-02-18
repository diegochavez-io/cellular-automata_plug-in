# Phase 3: Extract CASimulator - Research

**Researched:** 2026-02-17
**Domain:** Python refactoring — extract simulation class from pygame viewer, headless operation, class interface design
**Confidence:** HIGH

## Summary

This phase is a pure Python refactoring task. No new external libraries are needed. The job is to lift the simulation logic that currently lives inside the `Viewer` class into a standalone `CASimulator` class in `simulator.py`, with a contract that the resulting import chain is completely pygame-free.

The good news: the engine files (`lenia.py`, `smoothlife.py`, `mnca.py`, `gray_scott.py`, `engine_base.py`) already import nothing but numpy. The iridescent pipeline (`iridescent.py`) also has zero pygame imports. The problem is entirely contained in `viewer.py` and `controls.py`. The extraction boundary is already implied by the code — `viewer.py` holds both simulation state and display state, and disentangling them is straightforward because simulation state is clearly identifiable (everything that is not a `pygame.*` call or a UI interaction).

One subtlety: `viewer.py` imports `pygame` at the top of the module unconditionally. This means the `snap()` mode in `__main__.py` already does manual headless workarounds (duplicating the engine creation and containment logic inline). Phase 3 eliminates this duplication by making `CASimulator` the single authoritative headless path.

The `render_float()` addition to `IridescentPipeline` (SIM-04) is trivial: `return self.display_buffer.astype(np.float32) / 255.0`. The display buffer is already (H,W,3) uint8 populated by `render()`.

**Primary recommendation:** Move all Viewer `__init__` simulation state and the entire step/advect/contain/seed/reseed/LFO/coverage logic block into `CASimulator`. `Viewer` becomes an 80-line pygame wrapper that owns a `CASimulator` instance and calls it. The visual output must be pixel-identical because the same numpy computation runs — no behavioral change.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SIM-01 | CASimulator class in `simulator.py` encapsulating all simulation logic | See "Architecture Patterns" — exact attribute inventory and method signatures derived from current Viewer code |
| SIM-02 | `simulator.py` import chain is pygame-free; verified with `sys.modules['pygame'] = None` guard | See "Headless Guard" pattern and "Common Pitfalls / Import Contamination" |
| SIM-03 | Viewer delegates to CASimulator; `python -m cellular_automata coral` produces identical output | See "Viewer Delegation" pattern; identical output guaranteed when same numpy arrays are fed to same iridescent pipeline |
| SIM-04 | `IridescentPipeline.render_float()` returning `(H,W,3) float32 [0,1]` | Trivial one-liner on top of existing `render()` — see "render_float() Implementation" |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | Already present (float32 arrays everywhere) | All simulation math, containment masks, flow fields, noise pool | The entire CA stack is built on numpy — no change |
| scipy.ndimage | Already present (optional, guarded) | gaussian_filter, zoom, map_coordinates for advection | Already guarded with try/except in viewer.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| math | stdlib | `math.pi`, `math.sin`, `math.cos` in LFO systems | Already used by SinusoidalLFO and GS render |
| time | stdlib | Wall-clock dt in CASimulator for LFO update | Needed if Scope plugin doesn't supply dt (Phase 4 concern) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Lifting code directly from Viewer | Rewriting from scratch | Direct lift is safer — same numpy ops = same output, zero visual regression risk |
| One big `simulator.py` | Splitting into `simulator.py` + `lfo.py` | Keep LFO classes in `simulator.py` for Phase 3 scope; can split later if desired |

**Installation:** No new packages needed. All dependencies already in `requirements.txt`.

---

## Architecture Patterns

### Recommended Project Structure (after Phase 3)

```
cellular_automata/
├── simulator.py        # NEW: CASimulator + LFO classes (moved from viewer.py)
├── iridescent.py       # MODIFIED: add render_float()
├── viewer.py           # MODIFIED: thin pygame wrapper, delegates to CASimulator
├── __main__.py         # MODIFIED: snap() uses CASimulator instead of duplicated logic
├── engine_base.py      # UNCHANGED
├── lenia.py            # UNCHANGED
├── smoothlife.py       # UNCHANGED
├── mnca.py             # UNCHANGED
├── gray_scott.py       # UNCHANGED
├── presets.py          # UNCHANGED
├── smoothing.py        # UNCHANGED
├── controls.py         # UNCHANGED (pygame UI — stays in viewer layer)
└── __init__.py         # UNCHANGED (or add CASimulator export)
```

### Pattern 1: CASimulator Class Interface

**What:** A self-contained simulation object holding all non-display state and logic.
**When to use:** Whenever simulation state needs to be decoupled from the display loop.

Inventory of state that moves from `Viewer.__init__` into `CASimulator.__init__`:

```python
# simulator.py — ALL of this was previously in Viewer.__init__

class CASimulator:
    def __init__(self, preset_key="coral", sim_size=1024):
        # Resolution / scaling
        self.sim_size = sim_size
        self.res_scale = sim_size / 512  # BASE_RES = 512

        # Engine and preset state
        self.preset_key = preset_key
        self.engine_name = None
        self.engine = None  # CAEngine instance

        # Iridescent color pipeline
        self.iridescent = IridescentPipeline(sim_size)

        # LFO systems (engine-specific)
        self.lfo_system = None         # LeniaLFOSystem
        self.gs_lfo_system = None      # GrayScottLFOSystem
        self.sl_lfo_system = None      # SmoothLifeLFOSystem
        self.mnca_lfo_system = None    # MNCALFOSystem

        # Smoothed parameter infrastructure
        self.smoothed_params = {}      # key -> SmoothedParameter
        self.param_coupler = None      # LeniaParameterCoupler (Lenia only)

        # Pre-computed fields (geometry, only depends on sim_size)
        self._containment = ...        # (H,W) float32 radial decay
        self._noise_mask = ...         # (H,W) float32 gaussian
        self._noise_pool = [...]       # 6x (H,W) float32 noise frames
        self._noise_idx = 0
        self._color_offset = ...       # (H,W) float32 multi-octave
        self._cca_mask = ...           # (H,W) float32 circular
        self._mnca_containment = ...   # (H,W) float32 wide radial
        self._stir_dx, self._stir_dy = ...  # (H,W) float32 stir components
        self._stir_phase = 0.0
        self._flow_fields = ...        # dict of 7 (vx,vy) pairs
        self._flow_base_y = ...        # (H,W) float32 coordinate grids
        self._flow_base_x = ...
        self._flow_adv_y = ...         # (H,W) float32 pre-allocated
        self._flow_adv_x = ...

        # Flow strengths (set per-preset)
        self._flow_radial = 0.0
        self._flow_rotate = 0.0
        # ... (7 total)

        # Speed / fractional step accumulator
        self.sim_speed = 1.0
        self.speed_accumulator = 0.0
        self.render_thickness = 0.0

        # Coverage management state
        self._prev_mass = 0.0
        self._stagnant_frames = 0
        self._nucleation_counter = 0
        self._prev_world = None
        self._perturb_counter = 0

        self._apply_preset(preset_key)

    def apply_preset(self, key):
        """Public: switch preset (wraps _apply_preset)."""
        self._apply_preset(key)

    def set_runtime_params(self, **kwargs):
        """Accept runtime kwargs: preset, speed, hue, brightness, thickness, reseed."""
        # preset change: call apply_preset
        # speed, thickness: set attributes
        # hue: call iridescent.set_hue_offset()
        # brightness: set iridescent.brightness
        # reseed toggle: call engine.seed()

    def step(self, dt) -> np.ndarray:
        """Advance simulation by dt seconds. Returns (H,W,3) uint8."""
        # 1. Update smoothed params
        # 2. Update LFOs -> set engine params
        # 3. Fractional speed accumulator -> engine.step() loop
        # 4. After each step: advect, contain, stir/noise/perturb
        # 5. Coverage management (reseed, nucleation, anti-stagnation)
        # 6. Render frame -> return uint8 rgb
        ...

    def render_float(self, dt) -> np.ndarray:
        """Advance simulation and return (H,W,3) float32 [0,1]."""
        rgb_uint8 = self.step(dt)
        return rgb_uint8.astype(np.float32) / 255.0
```

Methods that move verbatim from Viewer into CASimulator:
- `_build_containment()`, `_build_noise_mask()`, `_build_stir_field()`
- `_build_color_offset()`, `_build_cca_mask()`, `_build_mnca_containment()`
- `_build_flow_fields()`, `_advect()`, `_fast_noise()`
- `_drop_center_seed()`, `_drop_seed_cluster()`, `_drop_center_seed_at()`
- `_scale_R()`, `_create_engine()`, `_apply_preset()` (as `_apply_preset`)
- `_render_frame()`, `_render_gs_emboss()`, `_apply_bloom()`
- `_rebuild_sim_fields()`
- All LFO class definitions: `SinusoidalLFO`, `LeniaLFOSystem`, `GrayScottLFOSystem`, `SmoothLifeLFOSystem`, `MNCALFOSystem`

### Pattern 2: Thin Viewer After Delegation

After extraction, `viewer.py` owns only display state:

```python
class Viewer:
    def __init__(self, width=900, height=900, sim_size=1024, start_preset="coral"):
        self.canvas_w = width
        self.canvas_h = height
        self.panel_visible = True
        self.running = True
        self.paused = False
        self.show_hud = True
        self.fullscreen = False
        self.brush_radius = 20
        self.fps_history = []

        # Delegate ALL simulation to CASimulator
        self.simulator = CASimulator(preset_key=start_preset, sim_size=sim_size)

        # UI (panel built after pygame.init in run())
        self.panel = None
        self.sliders = {}
        self.preset_buttons = None

    def run(self):
        pygame.init()
        # ... event loop, calls self.simulator.step(dt) each frame
        # ... gets surface via pygame.surfarray.make_surface(...)
        # ... panel callbacks call self.simulator.set_runtime_params(...)
```

What stays in `viewer.py`:
- `pygame.init()` and display setup
- `pygame.event` loop and keyboard handling
- `_handle_mouse()` (calls `self.simulator.engine.add_blob/remove_blob`)
- `_build_panel()` and panel callbacks (these call `simulator.set_runtime_params()`)
- `_draw_hud()` (reads `self.simulator.engine.stats`)
- `_save_screenshot()` (calls `self.simulator.step()` or uses current surface)
- The pygame surface creation: `pygame.surfarray.make_surface(rgb.swapaxes(0,1).copy())`

### Pattern 3: Headless Import Guard

The verification requirement (SIM-02) calls for this test:

```python
# Verified by: python -c "
import sys
sys.modules['pygame'] = None  # poison the import
from cellular_automata.simulator import CASimulator  # must not raise
sim = CASimulator('coral', 512)
frame = sim.render_float(0.016)
assert frame.shape == (512, 512, 3)
assert frame.dtype == np.float32
print('headless OK')
"
```

This works because `simulator.py` only imports:
- `math`, `time` (stdlib)
- `numpy` (no pygame dep)
- `.lenia`, `.smoothlife`, `.mnca`, `.gray_scott`, `.cca`, `.life`, `.excitable` (numpy only)
- `.iridescent` (numpy only)
- `.presets` (pure Python dicts)
- `.smoothing` (math + numpy)
- `scipy.ndimage` (guarded with try/except, already the pattern)

### Pattern 4: render_float() Implementation

```python
# In IridescentPipeline (iridescent.py):

def render_float(self, world, dt, lfo_phase=None, color_weights=None, t_offset=None):
    """Like render() but returns float32 [0,1] instead of uint8.

    Args: (same as render())
    Returns: (H, W, 3) float32 array in [0, 1]
    """
    rgb_uint8 = self.render(world, dt, lfo_phase=lfo_phase,
                            color_weights=color_weights, t_offset=t_offset)
    return rgb_uint8.astype(np.float32) / 255.0
```

Or alternatively (zero-copy path, uses pre-existing display_buffer):

```python
def render_float(self, world, dt, **kwargs):
    self.render(world, dt, **kwargs)  # populates self.display_buffer
    return self.display_buffer.astype(np.float32) / 255.0
```

The `astype(np.float32)` creates a new array (required — Scope plugin needs to own the data, `display_buffer` is reused each frame).

NOTE: The requirement spec says `render_float()` on `IridescentPipeline` but the full pipeline (including GS emboss, bloom, thickness dilation, engine-specific preprocessing) is in `Viewer._render_frame()`. The natural design is:
- `IridescentPipeline.render_float()` wraps the existing `render()` call — simple float conversion
- `CASimulator.render_float(dt)` calls the full frame render path (including GS emboss, bloom, etc.) and returns float32

This means `CASimulator.render_float(dt)` is what the Scope plugin actually calls. The `IridescentPipeline.render_float()` is a utility method that `CASimulator` may use internally (or not — it can just convert in `render_float(dt)`).

### Anti-Patterns to Avoid

- **Leaking simulation state into Viewer:** After extraction, `Viewer` must not hold any simulation fields directly (no `self.engine`, `self.lfo_system`, etc.). All access goes through `self.simulator.*`.
- **Splitting the LFO update loop:** The LFO update + engine `set_params()` + speed accumulator + advection + containment must remain an atomic sequence inside `CASimulator.step()`. Do not split across methods that could be called out of order.
- **Removing the scipy guard:** The `try/except ImportError` pattern for scipy imports must remain in `simulator.py` — same as current `viewer.py`. CuPy port in Phase 5 will add another guard on top.
- **Calling `viewer.py` from `simulator.py`:** Zero imports from viewer layer allowed in simulator. The dependency arrow must be: `viewer.py → simulator.py`, never the reverse.
- **Breaking `__main__.py` snap mode:** After Phase 3, the `snap()` function in `__main__.py` should be simplified to use `CASimulator` directly instead of its current duplicated inline engine creation code.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Float conversion | Custom quantization | `uint8_array.astype(np.float32) / 255.0` | Standard numpy idiom, zero ambiguity |
| Import-time pygame guard | `hasattr` checks | `sys.modules['pygame'] = None` before import (in tests) | Tests isolation — not runtime code |
| Headless simulation | Custom buffer-based pipeline | Move existing Viewer code verbatim | Same numpy ops = guaranteed identical output |

**Key insight:** This phase is pure mechanical refactoring, not a rewrite. The risk is in accidentally changing behavior during extraction (forgetting a state variable, reordering operations). Move code verbatim and test visual output before optimizing anything.

---

## Common Pitfalls

### Pitfall 1: Forgetting GrayScott's Dual-World Advection
**What goes wrong:** GrayScott has two concentration arrays (`engine.U` and `engine.V`). The advection code in `viewer.py` (lines 1579-1584) explicitly advects both and then sets `engine.world = engine.V`. If only `engine.world` is advected in `CASimulator.step()`, GS advection silently breaks.
**Why it happens:** All other engines use only `engine.world`. GS is the exception.
**How to avoid:** Copy the exact `elif self.engine_name == "gray_scott":` branch from `Viewer.run()` — advect `engine.U`, advect `engine.V`, clip both, then `engine.world = engine.V`.
**Warning signs:** GS patterns drift off-screen or become static after Phase 3.

### Pitfall 2: render_thickness Attribute Missing in CASimulator
**What goes wrong:** `_render_frame()` references `self.render_thickness`. If the attribute isn't initialized in `CASimulator.__init__`, it raises `AttributeError` only when a preset with `"thickness"` is loaded.
**Why it happens:** `render_thickness` is set in `_apply_preset()` from `preset.get("thickness", 0.0)`. If `_apply_preset` is called before `self.render_thickness` is initialized, the assignment works — but if `_render_frame()` is called before `_apply_preset`, it fails.
**How to avoid:** Initialize `self.render_thickness = 0.0` in `__init__` before calling `_apply_preset`.

### Pitfall 3: `_color_offset` is Randomized — Each Instance Gets Different Colors
**What goes wrong:** `_build_color_offset()` uses `np.random.randn(...)` to build spatial noise. In `Viewer`, this is called once at init. If `CASimulator` is re-instantiated (e.g., Scope plugin creates a new instance), the color zones will differ from the previous session.
**Why it happens:** This is by design (randomness makes each run unique), but if someone expects deterministic output across instances, they'll be surprised.
**How to avoid:** Document that `_color_offset` is intentionally random. For the Scope plugin, this is desirable behavior. No fix needed — just understand it.

### Pitfall 4: Panel Callbacks Still Call Viewer Attributes Directly
**What goes wrong:** In `_build_panel()`, lambda callbacks like `lambda v: setattr(self, 'render_thickness', v)` reference `self` (the Viewer). After Phase 3, these callbacks must be updated to set `self.simulator.render_thickness` or call `self.simulator.set_runtime_params(thickness=v)`.
**Why it happens:** The panel callbacks are closures that capture `self` from the Viewer context.
**How to avoid:** When moving `_build_panel()` to remain in Viewer, update all lambdas to go through `self.simulator`. The `_make_param_callback()`, `_make_flow_callback()`, speed/hue/brightness/thickness callbacks all need updating.

### Pitfall 5: `_hue_value` Viewer-Local State vs. Simulator State
**What goes wrong:** `Viewer` stores `self._hue_value = 0.25` and syncs it to `iridescent.set_hue_offset()`. After extraction, `_hue_value` belongs in the Viewer (it tracks the slider position), but `iridescent` is in `CASimulator`. The sync must go through `simulator.set_runtime_params(hue=v)`.
**Why it happens:** Slider state (what position the slider is at) lives in Viewer; effect (what the pipeline does) lives in CASimulator.
**How to avoid:** Keep `_hue_value` in Viewer for slider tracking. Panel callback calls `simulator.set_runtime_params(hue=val)`. `CASimulator.set_runtime_params()` calls `self.iridescent.set_hue_offset(val)`.

### Pitfall 6: `_prev_world` Not Included in State Transfer
**What goes wrong:** Velocity-driven perturbation (`velocity = np.abs(w - self._prev_world)`) requires `_prev_world`. If this isn't initialized to `None` in `CASimulator.__init__`, the perturbation branch silently skips or crashes.
**Why it happens:** `_prev_world` is easy to miss because it's initialized inline in the step loop (`if self._prev_world is None:`).
**How to avoid:** Explicitly initialize `self._prev_world = None` in `CASimulator.__init__`. This is already implicitly done by the viewer's `__init__` via `self._prev_world = None` — just make sure it's in the new class.

### Pitfall 7: Module-Level `ENGINE_CLASSES` Dict Needs to Move or be Imported
**What goes wrong:** `_create_engine()` references `ENGINE_CLASSES`, a module-level dict defined at the top of `viewer.py`. When this method moves to `simulator.py`, `ENGINE_CLASSES` must move with it (or be imported from somewhere shared).
**Why it happens:** It's a module-level constant, not a class attribute.
**How to avoid:** Move `ENGINE_CLASSES` dict to `simulator.py` or define it inline in `_create_engine()`. `viewer.py`'s `snap()` mode in `__main__.py` also imports `ENGINE_CLASSES` from `viewer` — update that import to use `simulator`.

### Pitfall 8: `FLOW_KEYS` and `FLOW_SLIDER_DEFS` Split
**What goes wrong:** `FLOW_KEYS` is used in both `_advect()` (simulation logic → goes to `simulator.py`) and `_build_panel()` / `_sync_sliders_from_engine()` (display logic → stays in `viewer.py`). If it only lives in one file, the other needs to import it.
**Why it happens:** Module-level constants shared between simulation and display code.
**How to avoid:** Define `FLOW_KEYS` in `simulator.py` (authoritative simulation side), import into `viewer.py` from `simulator`. Or duplicate the small list — it's 7 strings, duplication is fine.

---

## Code Examples

### Verified Pattern: Headless Guard Test

```python
# Run this after Phase 3 to verify SIM-02:
import sys
import numpy as np
sys.modules['pygame'] = None  # Poison the pygame import
# If simulator.py imports pygame (directly or transitively), this raises ImportError
from cellular_automata.simulator import CASimulator
sim = CASimulator("coral", 512)
frame = sim.render_float(0.016)
assert frame.shape == (512, 512, 3), f"Wrong shape: {frame.shape}"
assert frame.dtype == np.float32, f"Wrong dtype: {frame.dtype}"
assert 0.0 <= frame.min() and frame.max() <= 1.0, "Out of range"
print("SIM-02 PASS: pygame-free import chain confirmed")
```

### Verified Pattern: Existing Import Chain (pygame-free engines)

Confirmed by grep — these files import ONLY numpy and each other:
- `engine_base.py`: `abc`, `numpy`
- `lenia.py`, `smoothlife.py`, `mnca.py`, `gray_scott.py`: `numpy`, `.engine_base`
- `iridescent.py`: `numpy`
- `presets.py`: pure Python (no imports)
- `smoothing.py`: `math`, `numpy`

Only `viewer.py`, `controls.py`, and `__main__.py` (conditionally) import pygame.

### Verified Pattern: GrayScott Dual-World Advection (must be preserved)

```python
# From viewer.py lines 1579-1584 — this exact logic must move to CASimulator.step():
if self.engine_name == "gray_scott":
    self.engine.U = self._advect(self.engine.U)
    self.engine.V = self._advect(self.engine.V)
    np.clip(self.engine.U, 0.0, 1.0, out=self.engine.U)
    np.clip(self.engine.V, 0.0, 1.0, out=self.engine.V)
    self.engine.world = self.engine.V
elif self.engine_name in ("lenia", "smoothlife", "mnca"):
    self.engine.world = self._advect(self.engine.world)
```

### Verified Pattern: Full LFO → Engine Parameter Update Sequence (must be preserved as-is)

```python
# From viewer.py run() loop — this block must move verbatim into CASimulator.step():
# The order matters: smoothed params update → LFO reads smoothed → engine gets LFO output
if self.lfo_system:
    if "mu" in self.smoothed_params:
        self.lfo_system.mu_lfo.base_value = self.smoothed_params["mu"].get_value()
    if "sigma" in self.smoothed_params:
        self.lfo_system.sigma_lfo.base_value = self.smoothed_params["sigma"].get_value()
    if "T" in self.smoothed_params:
        base_T = self.smoothed_params["T"].get_value()
        self.lfo_system.T_lfo.base_value = base_T
        self.lfo_system.T_lfo.amplitude = base_T * 0.25
    self.lfo_system.update(dt)
    modulated = self.lfo_system.get_modulated_params()
    self.engine.set_params(**modulated)
# ... (similar blocks for GS, SL, MNCA LFO systems)
```

### Verified Pattern: render_float() on IridescentPipeline

```python
# Add to IridescentPipeline in iridescent.py:
def render_float(self, world, dt, lfo_phase=None, color_weights=None, t_offset=None):
    """Render and return (H, W, 3) float32 [0, 1].

    Thin wrapper over render() that converts uint8 output to float32.
    Creates a new array — caller owns the data.
    """
    rgb_uint8 = self.render(world, dt, lfo_phase=lfo_phase,
                            color_weights=color_weights, t_offset=t_offset)
    return rgb_uint8.astype(np.float32) / 255.0
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Simulation + display mixed in Viewer | CASimulator (headless) + Viewer (display) | Enables Scope plugin without pygame dependency |
| snap() in `__main__.py` duplicates engine creation logic | snap() calls CASimulator directly | Single source of truth for headless rendering |
| `iridescent.render()` returns uint8 only | `render_float()` added | Scope plugin gets clean float32 tensor without extra conversion |

---

## Open Questions

1. **Should `_render_frame()` / `_render_gs_emboss()` / `_apply_bloom()` be on `CASimulator` or on `IridescentPipeline`?**
   - What we know: These are currently on `Viewer` but have no pygame dependency. They only use numpy + scipy.
   - What's unclear: Whether to fold them into `IridescentPipeline` or leave them as private methods on `CASimulator`.
   - Recommendation: Leave on `CASimulator` for Phase 3. They compose simulation state (`engine_name`, `engine.world`, `_cca_mask`, `_color_offset`) with the iridescent pipeline — `CASimulator` is the right owner. `IridescentPipeline.render_float()` just wraps the existing `render()`.

2. **Should `CASimulator.step(dt)` call the full rendering pipeline or should `step()` only advance the simulation and a separate `render(dt)` call produce the frame?**
   - What we know: The requirement spec says `step(dt) → (H,W,3) uint8` and `render_float(dt) → (H,W,3) float32`. This implies both advance the sim AND render.
   - What's unclear: Whether the Scope plugin wants step-without-render capability.
   - Recommendation: Follow the spec as written. `step(dt)` advances + renders uint8, `render_float(dt)` advances + renders float32. For Phase 4 Scope plugin, only `render_float(dt)` is called. If step-without-render is needed later, split can happen then.

3. **`__main__.py` snap() function: full refactor or leave as-is?**
   - What we know: `snap()` currently duplicates engine creation, containment, and rendering inline. After Phase 3, it can be greatly simplified to use `CASimulator`.
   - Recommendation: Refactor `snap()` to use `CASimulator` in Phase 3. This reduces duplicate code, and the refactor is a good integration test that `CASimulator` works headlessly.

---

## Sources

### Primary (HIGH confidence)
- Direct code reading of `/Users/agi/Code/daydream_scope/plugins/cellular_automata/viewer.py` — full file, 1766 lines
- Direct code reading of `/Users/agi/Code/daydream_scope/plugins/cellular_automata/iridescent.py` — full file, 307 lines
- Direct code reading of `/Users/agi/Code/daydream_scope/plugins/cellular_automata/smoothing.py` — full file, 270 lines
- Direct code reading of `/Users/agi/Code/daydream_scope/plugins/cellular_automata/engine_base.py` — full file, 100 lines
- Direct code reading of `/Users/agi/Code/daydream_scope/plugins/cellular_automata/presets.py` — full file, 467 lines
- Direct code reading of `/Users/agi/Code/daydream_scope/plugins/cellular_automata/__main__.py` — full file, 256 lines
- `grep -rn "import pygame"` across all plugin files — confirmed only 3 files have pygame imports

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all deps already present and confirmed
- Architecture: HIGH — extraction boundary is unambiguous; simulation state vs display state is clear in the existing code
- Pitfalls: HIGH — all pitfalls are derived from direct reading of the actual code, not speculation

**Research date:** 2026-02-17
**Valid until:** Indefinite — this is codebase-specific research, not library-API research. Valid as long as the codebase structure doesn't change.
