# Feature Landscape: Scope Plugin Deployment Milestone

**Domain:** Cellular Automata Video Source — Scope Plugin + GPU Acceleration + RunPod Deployment
**Researched:** 2026-02-17
**Confidence:** HIGH (first-party source: example_plugin/ lives in this repo; code analysis of CA engines; CLAUDE.md authoritative on RunPod setup)

---

## Context: What This Milestone Is

Phase 1 (local prototype) is done. The CA runs beautifully in pygame. This milestone takes it from
"pygame app" to "Scope pipeline that outputs AI-enhanced video on RunPod".

Three sequential phases, each depending on the previous:

```
Phase A: Scope Plugin Wrapper  (CA runs headless inside Scope)
    |
Phase B: CuPy GPU Acceleration (CA hits 30+ FPS at 1024x1024 on GPU)
    |
Phase C: RunPod Deployment     (Full pipeline: CA → Scope → Krea Realtime)
```

The existing CA code is the engine. The work is plumbing, not new CA features.

---

## Table Stakes

Features that must work for the milestone to be considered done at all.

| Feature | Why Non-Negotiable | Complexity | Dependency |
|---------|-------------------|------------|-----------|
| **Headless engine execution** | pygame display cannot exist on a headless server; Scope is the display | LOW | Prerequisite for all Scope work |
| **THWC float tensor output** | Scope contract: `__call__()` must return `torch.Tensor` shape `(T, H, W, C)` in `[0, 1]` | LOW | Existing `rgb uint8` must become `float32 / 255.0` |
| **`register_pipelines` hook** | How Scope discovers the plugin at all; without it nothing loads | LOW | Follows example_plugin/plugin.py pattern exactly |
| **`pyproject.toml` entry point** | `[project.entry-points."scope"]` wires the plugin to Scope's loader | LOW | One line in pyproject.toml |
| **Runtime kwargs for preset/speed/hue** | Scope UI passes runtime params via `**kwargs` in `__call__()` — not stored in `__init__()` | MEDIUM | Critical: storing in `__init__` breaks live UI control |
| **`uv run daydream-scope install` works** | Installation command must succeed before any testing | LOW | pyproject.toml + hatchling build backend |
| **Continuous frame output** | Scope calls `__call__()` on a cadence; must always return valid frames, never hang | MEDIUM | CA must not block on pygame events |
| **No pygame dependency in plugin** | pygame-ce is a dev/viewer dependency, not a plugin dependency | LOW | Plugin imports engines directly, not `viewer.py` |

---

## Differentiators

Features that make the CA pipeline more useful/controllable than a static video source.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Preset selector in Scope UI** | Switch between Lenia/SmoothLife/MNCA/GS aesthetics without restarting Scope | MEDIUM | Requires engine swap on preset change; must handle state transition cleanly |
| **Live speed control** | Adjust simulation tempo from Scope UI while CA runs | LOW | Read `speed` from kwargs each frame; scale step count |
| **Live hue shift** | Recolor the organism from Scope UI without reseed | LOW | Pass `hue_override` to IridescentPipeline.render() |
| **Hot reseed button** | Restart organism from Scope UI without reloading pipeline | LOW | Expose as a boolean kwarg flag; call `engine.seed()` |
| **CuPy GPU acceleration** | Hit 30+ FPS at 1024x1024 vs current ~12-18 FPS (CPU) — enables real-time Scope output | HIGH | Requires numpy → CuPy port per engine |
| **Numpy/CuPy auto-detection** | Plugin runs on CPU (local) and GPU (RunPod) without code changes | LOW | `try: import cupy as np; except: import numpy as np` pattern |
| **Coexists with 14B model in VRAM** | RTX 5090 has 32GB; CA state at 1024x1024 float32 = ~4MB; FFT buffers ~20MB total — negligible | LOW | No special handling needed; CuPy allocates on demand |

---

## Anti-Features

Features that seem appealing but create real problems for this milestone. Do not build these.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Pygame window inside Scope** | Scope is the UI; launching a pygame window alongside it causes two event loops fighting over display | Remove all `pygame.init()` / `pygame.display` calls from the plugin path; import engines directly |
| **Exposing raw CA parameters (mu, sigma, R, T) in Scope UI** | These parameters crash the organism if set wrong; they are not meaningful to a Scope user | Expose only: preset name, speed multiplier, hue shift. Advanced users can edit presets in code. |
| **Streaming individual CA frames as single-frame tensors** | THWC with T=1 is valid but wasteful; Scope may expect more temporal context for its diffusion model | Return T=4 frames per call (4 consecutive CA steps), giving Krea Realtime motion information |
| **Per-call engine reinitialization** | Rebuilding FFT kernels on every `__call__()` is 50-200ms of overhead — kills real-time | Initialize engine in `__init__()`; only rebuild on preset change |
| **scipy.ndimage in the hot path on GPU** | CuPy has `cupyx.scipy.ndimage` but mixing CPU scipy + GPU cupy arrays requires device transfers | Port all hot-path scipy calls (gaussian_filter, map_coordinates, zoom) to CuPy equivalents |
| **Full viewer.py in the plugin** | viewer.py has 1000+ lines of pygame UI code; importing it pulls in pygame as a dependency | Create a thin `ca_pipeline.py` that imports only: engine classes, IridescentPipeline, presets |
| **Saving screenshots inside Scope pipeline** | PIL/pygame file I/O in `__call__()` blocks the pipeline loop | Remove screenshot code from plugin path entirely |
| **Multiple Scope pipelines for each CA engine** | Registering "Lenia Pipeline", "GS Pipeline", etc. creates clutter in Scope's UI | One pipeline: "Cellular Automata" with a preset dropdown that switches engines internally |

---

## Feature Dependencies

```
PHASE A: Scope Plugin Wrapper
    Headless engine extraction
        requires: engine_base.py, lenia.py, smoothlife.py, mnca.py, gray_scott.py
        requires: iridescent.py (rendering pipeline)
        requires: presets.py (preset definitions)
        excludes: viewer.py, controls.py, pygame
        |
    ca_pipeline.py (new file)
        requires: headless engine extraction
        requires: THWC tensor output format
        requires: runtime kwargs (preset, speed, hue)
        |
    plugin.py (new file for CA plugin)
        requires: ca_pipeline.py
        requires: register_pipelines hook
        |
    pyproject.toml (new for CA plugin)
        requires: plugin.py
        requires: hatchling build backend
        |
    uv run daydream-scope install (test)
        requires: pyproject.toml

PHASE B: CuPy GPU Acceleration
    depends on: Phase A complete and tested
    |
    numpy/cupy shim (xp = cupy or numpy)
        requires: Phase A pipeline working on CPU
        |
    engine_base.py CuPy port
        requires: xp shim
        world array moves to GPU
        |
    lenia.py CuPy port (FFT convolution)
        requires: engine_base CuPy
        np.fft.rfft2 -> cupy.fft.rfft2 (drop-in)
        |
    mnca.py CuPy port (FFT convolution)
        requires: engine_base CuPy
        same FFT pattern as Lenia
        |
    smoothlife.py CuPy port (FFT convolution)
        requires: engine_base CuPy
        same FFT pattern
        |
    gray_scott.py CuPy port (finite differences)
        requires: engine_base CuPy
        np.roll -> cupy.roll (drop-in)
        _laplacian pad+slice: pad/slice ops are CuPy-compatible
        |
    viewer.py flow field advection (map_coordinates)
        scipy.ndimage.map_coordinates -> cupyx.scipy.ndimage.map_coordinates
        CRITICAL: this is the most expensive CPU op in viewer.py
        |
    iridescent.py CuPy port (color pipeline)
        np.zeros -> cupy.zeros for GPU color buffers
        uint8 output: cupy.asnumpy() once per frame (device→host copy)
        |
    benchmark: verify 30+ FPS at 1024x1024

PHASE C: RunPod Deployment
    depends on: Phase B complete
    |
    CuPy installation in container
        requires: CUDA version match (RTX 5090 = CUDA 12.x)
        cupy-cuda12x pip package
        |
    LoRA coexistence test
        requires: CuPy plugin loaded alongside Krea Realtime
        verify VRAM headroom: CA ~20MB vs 14B model ~28GB
        |
    Plugin deployed via SSH to RunPod
        requires: croc or direct SSH file transfer
        uv run daydream-scope install on RunPod
        |
    End-to-end pipeline test
        CA frames -> Scope -> Krea Realtime + LoRA -> output video
```

---

## MVP Definition

### Phase A Launch Criteria (Scope Plugin)

- [ ] `uv run daydream-scope install /path/to/ca_plugin` succeeds
- [ ] "Cellular Automata" appears in Scope pipeline selector
- [ ] Plugin generates frames continuously (no hang, no crash)
- [ ] Preset selector in Scope UI changes CA behavior live
- [ ] Speed control works from Scope UI
- [ ] Hue shift works from Scope UI
- [ ] No pygame import in plugin path

### Phase B Launch Criteria (CuPy)

- [ ] Plugin auto-detects GPU and uses CuPy when available
- [ ] Falls back to numpy on CPU without code changes
- [ ] 30+ FPS at 1024x1024 on RTX 5090 (benchmark target)
- [ ] All 4 engines (Lenia, SmoothLife, MNCA, GS) run on GPU
- [ ] No per-frame device transfers except final uint8 readback

### Phase C Launch Criteria (RunPod)

- [ ] Plugin installed on RunPod pod
- [ ] CA pipeline appears in Scope running on RunPod
- [ ] CA + Krea Realtime run simultaneously without OOM
- [ ] Visual output reaches acceptable quality (Scope AI transform visible)
- [ ] Latency acceptable for intended use

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Phase | Priority |
|---------|------------|---------------------|-------|----------|
| Headless engine execution | HIGH | LOW | A | P1 |
| THWC tensor output | HIGH | LOW | A | P1 |
| `register_pipelines` + pyproject.toml | HIGH | LOW | A | P1 |
| Runtime kwargs (preset/speed/hue) | HIGH | MEDIUM | A | P1 |
| Continuous frame output (no hang) | HIGH | MEDIUM | A | P1 |
| Preset selector in Scope UI | HIGH | MEDIUM | A | P1 |
| Hot reseed button | MEDIUM | LOW | A | P2 |
| Numpy/CuPy auto-detect shim | HIGH | LOW | B | P1 |
| Engine FFT ports to CuPy | HIGH | MEDIUM | B | P1 |
| map_coordinates to cupyx | HIGH | MEDIUM | B | P1 |
| iridescent.py CuPy port | HIGH | LOW | B | P1 |
| Benchmark 30+ FPS | HIGH | LOW | B | P1 |
| CuPy in RunPod container | HIGH | LOW | C | P1 |
| LoRA coexistence test | HIGH | LOW | C | P1 |
| End-to-end pipeline test | HIGH | LOW | C | P1 |
| MIDI/OSC control | LOW | HIGH | future | P3 |
| Multi-CA-instance presets | LOW | HIGH | future | P3 |
| Web UI for CA parameters | LOW | HIGH | future | P3 |

---

## Scope Plugin API: Verified Facts

From `/Users/agi/Code/daydream_scope/plugins/example_plugin/pipeline.py` (first-party code, HIGH confidence):

### `__call__()` Return Contract
```python
# Return THWC tensor, values in [0, 1]
output = torch.rand(num_frames, height, width, channels)  # shape: (T, H, W, 3)
return output
```

The CA currently produces `np.ndarray` of shape `(H, W, 3)` in `uint8`. Conversion:
```python
rgb_uint8 = iridescent.render(world, dt, t_offset=color_offset)  # (H, W, 3) uint8
# Wrap in time dim, convert to float, normalize
frame_tensor = torch.from_numpy(rgb_uint8).float().unsqueeze(0) / 255.0  # (1, H, W, 3)
```

### `ui_field_config()` Structure
```python
@staticmethod
def ui_field_config():
    return {
        "preset_name": {
            "order": 1,
            "panel": "controls",
            "label": "Preset",
            # Scope will need to know this is a string selector —
            # verify whether Scope supports dropdown/enum fields
        },
        "speed": {
            "order": 2,
            "panel": "controls",
            "label": "Speed",
            "min": 0.5,
            "max": 3.0,
            "step": 0.1
        },
        "hue_shift": {
            "order": 3,
            "panel": "controls",
            "label": "Hue Shift",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01
        },
        "reseed": {
            "order": 4,
            "panel": "controls",
            "label": "Reseed"
            # boolean trigger — verify if Scope supports button-type params
        }
    }
```

**Open question:** Does Scope's `ui_field_config()` support string enum / dropdown fields for preset selection? The example only shows numeric sliders. If not, expose preset as an integer index (0-10) mapped to UNIFIED_ORDER internally.

### Runtime Parameter Access Pattern
```python
def __call__(self, prompt: str, **kwargs):
    # Read ALL runtime params from kwargs — not from self
    preset_name = kwargs.get("preset_name", "coral")
    speed = kwargs.get("speed", 1.0)
    hue_shift = kwargs.get("hue_shift", 0.0)
    reseed = kwargs.get("reseed", False)
    # ... generate and return THWC tensor
```

---

## CuPy Migration: Feature Map

All numpy operations used in the hot path, with CuPy equivalents.

| Operation | Location | numpy | CuPy | Notes |
|-----------|----------|-------|------|-------|
| FFT convolution | lenia.py, mnca.py, smoothlife.py | `np.fft.rfft2`, `np.fft.irfft2` | `cupy.fft.rfft2`, `cupy.fft.irfft2` | Drop-in; CuPy FFT uses cuFFT |
| Array ops (`clip`, `roll`, etc.) | all engines | `np.clip`, `np.roll` | `cupy.clip`, `cupy.roll` | Drop-in |
| Laplacian stencil (pad+slice) | gray_scott.py | `np.pad`, slice ops | `cupy.pad`, same slices | Drop-in |
| Bilinear interpolation (flow) | viewer.py flow field | `scipy.ndimage.map_coordinates` | `cupyx.scipy.ndimage.map_coordinates` | Must use cupyx variant; don't mix CPU scipy + GPU arrays |
| Gaussian blur (viewer) | viewer.py `_blur_world`, `_dilate_world` | `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` | Used in color pipeline; same pattern |
| Zoom/upsample | viewer.py `_blur_world` | `scipy.ndimage.zoom` | `cupyx.scipy.ndimage.zoom` | Used for bloom |
| Color pipeline (multiply, add) | iridescent.py | numpy array ops | cupy array ops | Drop-in; final `.get()` once for uint8 |
| Random noise pool | viewer.py | `np.random.randn` | `cupy.random.randn` | Pre-compute on GPU |
| Containment mask (static) | viewer.py | `np.ogrid`, `np.exp` | Compute once on GPU, reuse | Pre-built at init |

### Device Transfer Strategy (CRITICAL)
- Engine `world` array: lives on GPU (cupy)
- All per-step ops: on GPU (no transfers during step)
- Only transfer: `rgb_uint8 = cupy.asnumpy(gpu_rgb)` once per frame for host→tensor conversion
- Torch tensor: create from numpy on CPU, or use `torch.as_tensor(cupy_array, device='cuda')` to stay on GPU

### CuPy Install Note
RTX 5090 requires CUDA 12.x. Use `cupy-cuda12x` not `cupy-cuda11x`.
```bash
pip install cupy-cuda12x
```

---

## RunPod Deployment: Feature Map

| Feature | How It Works | Notes |
|---------|-------------|-------|
| Plugin installation on RunPod | `uv run daydream-scope install /path/to/ca_plugin` | Transfer plugin files via croc first |
| CuPy on RunPod | `pip install cupy-cuda12x` in container | CUDA already available on RunPod GPU pods |
| LoRA + CA coexistence | 14B model uses ~28GB VRAM; CA uses ~20MB; 32GB RTX 5090 has 4GB headroom | Tight but should work; monitor with `nvidia-smi` |
| SCOPE_LORA_DIR | `/workspace/models/lora/` — set by RunPod container env | Already documented; CA plugin does not use LoRAs directly |
| Plugin verification | Check Scope UI shows "Cellular Automata" in pipeline list | SSH to RunPod, check Scope logs |

---

## Open Questions

1. **Scope dropdown/enum support:** Does `ui_field_config()` support non-numeric fields (string enum for preset selection)? If not, use int index 0-N mapped to UNIFIED_ORDER. Verify by checking any other Scope plugins that exist, or test experimentally.

2. **Scope call cadence:** Does Scope call `__call__()` once per display frame (60fps) or once per generation step? If 60fps, the CA needs to decouple its step rate from Scope's call rate (run N steps per call based on speed param).

3. **THWC temporal dimension:** What T value does Krea Realtime expect? T=1 is valid per the contract but Krea Realtime may give better motion output with T=4+ consecutive frames. Test both.

4. **VRAM allocation timing:** Does CuPy allocate VRAM eagerly (at import) or lazily (at first use)? If eager, there may be an ordering dependency with Krea Realtime model loading. Test on RunPod.

5. **cupyx.scipy.ndimage availability:** Is `cupyx.scipy.ndimage.map_coordinates` a drop-in for `scipy.ndimage.map_coordinates` with CuPy arrays? Specifically: does it support `order=1, mode='wrap'`? These are the exact args used in flow field advection. Verify against CuPy docs before committing.

---

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Scope Plugin API | HIGH | example_plugin/ is first-party code in this repo; contract is clear |
| CuPy numpy equivalence | HIGH (with flags) | CuPy is well-established as numpy drop-in; cupyx.scipy.ndimage is less certain — flag open question #5 |
| RunPod deployment mechanics | HIGH | CLAUDE.md documents exact SSH/croc/path setup from working sessions |
| VRAM coexistence | MEDIUM | Math says it should fit (28GB model + 20MB CA); untested in practice |
| Scope UI field types (dropdown) | LOW | Only numeric examples seen; string enum support unverified |
| Krea Realtime T-dimension preference | LOW | No documentation found; must test experimentally |

---

## Sources

- `/Users/agi/Code/daydream_scope/plugins/example_plugin/pipeline.py` — Scope plugin API contract (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/plugin.py` — `register_pipelines` hook (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/pyproject.toml` — entry point format (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/cellular_automata/` — full CA codebase, all engines (HIGH confidence, first-party)
- `/Users/agi/Code/daydream_scope/CLAUDE.md` — RunPod setup, LoRA paths, SSH mechanics (HIGH confidence, battle-tested)
- CuPy numpy compatibility: training knowledge (MEDIUM; cupyx.scipy.ndimage flagged as LOW until verified)
