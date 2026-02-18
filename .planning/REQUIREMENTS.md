# Requirements: DayDream Scope — Cellular Automata Plugin

**Defined:** 2026-02-16
**Core Value:** Beautiful, constantly-evolving organic imagery that feeds Scope's pipeline — every system alive, iridescent, never black.

## v1 Requirements (COMPLETE)

Requirements for local polish milestone. Phases 1-2 complete, phases 3-6 deprioritized.

### Color System

- [x] **CLR-01**: Iridescent oil-slick shimmer renders across organism surface using thin-film interference model
- [x] **CLR-02**: Slow global hue sweep cycles entire organism through rainbow colors over time
- [x] **CLR-03**: Spatial gradient produces different colors across different parts of the organism body (prism effect)
- [x] **CLR-04**: RGB tint sliders (R, G, B) control overall color balance of the output
- [x] **CLR-05**: Brightness/darkness slider controls output luminance
- [x] **CLR-06**: All simulation engines render through the unified iridescent color pipeline
- [x] **CLR-07**: Old 4-layer Core/Halo/Spark/Memory system is removed

### LFO / Oscillator

- [x] **LFO-01**: LFO breathing is smooth sinusoidal — no sudden snap-back or reset
- [x] **LFO-02**: LFO speed slider on UI controls the breathing/undulation tempo
- [x] **LFO-03**: Default LFO cycle is very slow — organic, imperceptible growth and retreat

### Parameter Safety (DEPRIORITIZED — Scope UI replaces pygame controls)

- [ ] **SAF-01**: All engine parameter sliders use EMA smoothing (0.3-0.5s time constant)
- [ ] **SAF-02**: Abruptly moving any slider does not kill the organism
- [ ] **SAF-03**: Parameter ranges are clamped to safe bounds per engine

### Control Panel (DEPRIORITIZED — Scope UI replaces pygame controls)

- [ ] **UI-01**: Simplified control panel
- [ ] **UI-02**: Old layer weight sliders removed from UI
- [ ] **UI-03**: Preset selector shows all available presets

### Presets (DEPRIORITIZED)

- [ ] **PRE-01** through **PRE-04**: Preset cleanup

### New Engines (DEPRIORITIZED)

- [ ] **ENG-01** through **ENG-07**: Physarum, DLA, agent-based, primordial particles, stigmergy

---

## v2.0 Requirements — Scope Deployment

**Milestone:** Deploy the CA video source as a Scope plugin on RunPod with GPU acceleration.
**Created:** 2026-02-17
**Status:** Active

### Category 1: CASimulator Extraction

Extract simulation logic from `viewer.py` into a clean `CASimulator` class for headless operation.

- [x] **SIM-01: CASimulator Class** — Create `simulator.py` with `CASimulator` encapsulating all simulation logic from `viewer.py`: engine lifecycle, LFO systems, flow fields, advection, containment, noise/stir, coverage management, seeding, color rendering. Interface: `__init__(preset_key, sim_size)`, `apply_preset(key)`, `set_runtime_params(**kwargs)`, `step(dt) → (H,W,3) uint8`, `render_float(dt) → (H,W,3) float32 [0,1]`.

- [x] **SIM-02: Headless Operation** — `simulator.py` never imports pygame. Entire import chain (simulator → engines → iridescent → presets → smoothing) must be pygame-free. Verified with `sys.modules['pygame'] = None` guard before import.

- [x] **SIM-03: Viewer Delegation** — Refactor `viewer.py` to delegate simulation to `CASimulator`. Viewer becomes thin pygame display wrapper. `python -m cellular_automata coral` produces identical visual output.

- [x] **SIM-04: render_float()** — Add `render_float()` to `IridescentPipeline` returning `(H,W,3) float32 [0,1]` (uint8 / 255.0).

### Category 2: Scope Plugin Wrapper

Wrap CASimulator as a Scope text-only pipeline plugin.

- [ ] **PLUG-01: Plugin Registration** — `plugin.py` with `@hookimpl` decorated `register_pipelines(register)` from `scope.core.plugins.hookspecs`. Calls `register(CAPipeline)`.

- [ ] **PLUG-02: Pipeline Config Schema** — Pydantic config inheriting `BasePipelineConfig`: `pipeline_id="cellular-automata"`, `pipeline_name="Cellular Automata"`, `supports_prompts=False`, `modes={"text": ModeDefaults(default=True)}`. Load-time field: `sim_size`. Runtime fields: `preset` (enum), `speed` (slider), `hue` (slider), `brightness` (slider), `thickness` (slider), `reseed` (toggle). Each field uses `json_schema_extra=ui_field_config(...)`.

- [ ] **PLUG-03: Pipeline Class** — `CAPipeline` inheriting `scope.core.pipelines.interface.Pipeline`. Implements `get_config_class()` and `__call__(**kwargs) → {"video": tensor}`. Creates CASimulator in `__init__()`.

- [ ] **PLUG-04: Runtime kwargs** — ALL user-controllable params read from `kwargs` in `__call__()` every frame. Never stored in `__init__()`. Preset changes trigger `simulator.apply_preset()`.

- [ ] **PLUG-05: THWC Tensor Output** — Returns `{"video": tensor}` where tensor is `torch.Tensor (1, H, W, 3) float32 [0,1]`. Conversion: `render_float()` → `.copy()` → `torch.from_numpy()` → `.unsqueeze(0)`.

- [ ] **PLUG-06: Wall-Clock dt** — Track elapsed time between `__call__()` using `time.perf_counter()`. Pass dt to CASimulator for rate-independent LFO/speed. Clamp to `[0.001, 0.1]`.

- [ ] **PLUG-07: pyproject.toml** — Entry point `[project.entry-points."scope"]` → `cellular_automata.plugin`. Dependencies: numpy, scipy, torch. NO pygame. Build: hatchling. Excludes viewer.py, controls.py, __main__.py from wheel.

- [ ] **PLUG-08: Warmup Strategy** — Engine warmup deferred out of `__init__()` (first `__call__()` or background thread). Plugin loads instantly.

- [ ] **PLUG-09: Install and Test** — `uv run daydream-scope install .` succeeds. Pipeline appears in Scope UI. Slider controls update output live.

- [ ] **PLUG-10: No pygame in Import Chain** — plugin.py → pipeline.py → simulator.py → engines/iridescent/presets/smoothing: zero pygame imports. viewer.py and controls.py excluded from installed package.

### Category 3: CuPy GPU Acceleration

Port simulation hot path to CuPy for GPU operation.

- [ ] **GPU-01: Backend Module** — `backend.py` with conditional numpy/CuPy import. Exports `xp` and `xp_ndimage`. All engine/simulator code uses backend instead of direct numpy.

- [ ] **GPU-02: Engine GPU Port** — All 4 engines use `xp` from backend for FFT, array ops, random. Drop-in replacement, zero duplicate code paths.

- [ ] **GPU-03: Custom Bilinear Wrap** — CuPy bilinear interpolation with `cp.mod()` for advection. Replaces `map_coordinates(mode='wrap')`. Eliminates cupyx version risk.

- [ ] **GPU-04: CPU Color Pipeline** — IridescentPipeline stays on CPU. One `cp.asnumpy()` per frame (~0.3ms at 512x512). LUT is 192KB, cache-bound.

- [ ] **GPU-05: VRAM Budget** — 512x512 on GPU. CuPy memory pool capped at 512MB. Static fields built on CPU, transferred once.

- [ ] **GPU-06: Static Fields on CPU** — Containment, color offset, noise pool, flow fields built with scipy on CPU. Transferred to GPU with `cp.asarray()`. No cupyx.scipy.ndimage.zoom dependency.

- [ ] **GPU-07: CPU/GPU Auto-Toggle** — Auto-detect CuPy at init. Clear logging of backend choice. Works correctly on both CPU (local) and GPU (RunPod).

- [ ] **GPU-08: Performance Target** — 30+ FPS at 512x512 on RTX 5090. All 4 engines survive 500+ GPU steps. Visual QA on MNCA thresholds and GS dead zone.

### Category 4: RunPod Deployment

Deploy CA plugin to RunPod alongside Krea Realtime.

- [ ] **DEPL-01: Container Setup** — Install `cupy-cuda12x` and CA plugin in Scope container on RunPod. Verify CUDA compatibility.

- [ ] **DEPL-02: Plugin Installation** — CA plugin installed in Scope on RunPod. "Cellular Automata" visible in pipeline selector.

- [ ] **DEPL-03: VRAM Coexistence** — CA + 14B Krea Realtime model coexist on RTX 5090 (32GB). CA < 200MB VRAM. No OOM.

- [ ] **DEPL-04: End-to-End Pipeline** — CA → Scope → Krea Realtime + LoRAs → AI-transformed video. Transform visibly applied.

- [ ] **DEPL-05: Performance Validation** — Acceptable latency for real-time output. No dropped frames or hangs.

---

## Summary

| Milestone | Category | Requirements | Status |
|-----------|----------|-------------|--------|
| v1.0 | Color + LFO | CLR-01–07, LFO-01–03 | Complete |
| v1.0 | Safety/UI/Presets/Engines | SAF, UI, PRE, ENG | Deprioritized |
| v2.0 | CASimulator Extraction | SIM-01–04 | Pending → Phase 3 |
| v2.0 | Scope Plugin Wrapper | PLUG-01–10 | Pending → Phase 4 |
| v2.0 | CuPy GPU Acceleration | GPU-01–08 | Pending → Phase 5 |
| v2.0 | RunPod Deployment | DEPL-01–05 | Pending → Phase 6 |

**v2.0 Total: 27 requirements across 4 phases**

---
*Created: 2026-02-16*
*Updated: 2026-02-17 — Added v2.0 Scope Deployment requirements*
