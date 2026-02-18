# Project Research Summary

**Project:** DayDream Scope — Cellular Automata Scope Plugin (Deployment Milestone)
**Domain:** Pygame/numpy CA simulation wrapped as a headless Scope plugin with GPU acceleration
**Researched:** 2026-02-17
**Confidence:** HIGH (Scope plugin API from first-party code; CuPy: MEDIUM on precision/compat; RunPod: HIGH from battle-tested sessions)

## Executive Summary

This milestone converts a working pygame/numpy cellular automata simulation into a DayDream Scope text-only pipeline that feeds live frames to Krea Realtime on RunPod. The core CA engines (Lenia, SmoothLife, MNCA, Gray-Scott) are pure numpy with no pygame dependency — they are ready to use as-is. The work is entirely plumbing: extract simulation logic from `viewer.py` into a clean `CASimulator` class, wrap it in a Scope-compliant `CAPipeline`, port the hot path to CuPy for GPU acceleration, and deploy on the RTX 5090 pod alongside the 14B model.

The recommended approach is strictly sequential across three phases with hard dependencies between them. Phase 1 (extract `CASimulator`) is the prerequisite for everything else: it enables headless operation, tests the Scope plugin contract, and creates the clean import boundary that CuPy porting requires. Phase 2 wires the plugin to Scope using the well-documented entry point pattern from `example_plugin/`. Phase 3 ports the FFT-heavy engine loop to CuPy using a thin backend shim that switches between numpy and CuPy with no duplicate code paths. Phase 4 deploys to RunPod and validates the end-to-end pipeline.

The key risks are: (1) the VRAM budget is tight — the 14B model occupies ~28GB of the RTX 5090's 32GB, so the CA plugin must run at 512x512 on GPU and cap its CuPy memory pool to 512MB; (2) `cupyx.scipy.ndimage.map_coordinates` with `mode='wrap'` has uncertain version support — implement a custom bilinear wrap fallback in CuPy; (3) the existing `Viewer` class stores all runtime state in `__init__()`, which is the opposite of the Scope plugin contract — all user-controllable params must come from `kwargs` in `__call__()`. Getting the architecture right in Phase 1 avoids a complete plugin rewrite later.

## Key Findings

### Recommended Stack

The existing stack (Python 3.10+, pygame-ce, numpy, scipy) is unchanged for local development. Three additions power the deployment milestone: `torch>=2.0.0` (already in Scope's base container) converts numpy frames to THWC tensors; `cupy-cuda12x>=13.3.0` provides the drop-in GPU replacement for numpy and scipy.ndimage ops; `hatchling` (the build backend used by `example_plugin/`) handles packaging. No new frameworks, no parallelism libraries, no custom CUDA kernels.

**Core technologies:**
- `torch>=2.0.0`: THWC tensor output required by Scope contract — already in `daydreamlive/scope:main` base image
- `cupy-cuda12x>=13.3.0`: Drop-in numpy replacement (CuPy 14.0.0 latest as of 2026-02-17); use `cupy-cuda12x` not `cupy-cuda11x` — RTX 5090 requires CUDA 12.6+
- `hatchling`: Build backend matching the example plugin toolchain; resolved automatically by `uv`
- `uv run daydream-scope install <path>`: The documented install path; handles entry point registration correctly

**Critical constraint:** `pygame-ce` must NOT be a plugin dependency. The `daydreamlive/scope:main` container has no display server. Importing pygame in any plugin-reachable module crashes the plugin on load with an SDL error.

### Expected Features

The milestone has no new CA simulation features — the organism behavior is locked from Phase 1. The work is integration features: making the CA controllable from Scope's UI and performant on GPU.

**Must have (table stakes):**
- Headless engine execution — no pygame in the plugin import chain
- THWC float tensor output `(1, H, W, 3)` in `[0, 1]` — the Scope pipeline contract
- `register_pipelines` hook + `pyproject.toml` entry point — how Scope discovers the plugin
- Runtime kwargs for preset/speed/hue — read from `**kwargs` in `__call__()` every frame
- Continuous frame output with no hangs or crashes
- No pygame dependency in plugin `pyproject.toml`

**Should have (differentiators):**
- Preset selector in Scope UI — switch between all 11 CA presets live
- Live speed and hue controls from Scope UI
- Hot reseed button (boolean kwarg trigger)
- numpy/CuPy auto-detection shim — same plugin code runs on CPU locally and GPU on RunPod
- 30+ FPS at 512x512 on RTX 5090

**Defer (post-milestone):**
- MIDI/OSC control
- Multi-CA-instance presets
- Web UI for raw CA parameters (mu, sigma, R, T — dangerous to expose without safety guards)
- Additional CA engines

**Open question — Scope UI enum support:** Does `ui_field_config()` support non-numeric string enum fields for the preset dropdown? Only numeric sliders are confirmed from `example_plugin/`. Fallback: expose preset as integer index 0–N mapped to `UNIFIED_ORDER` internally.

### Architecture Approach

The architecture centers on extracting a `CASimulator` class from `viewer.py` that holds all simulation logic (LFO systems, flow fields, containment, noise pool, advection, rendering) without any pygame dependency. Both the pygame `Viewer` (local interactive mode) and the Scope `CAPipeline` (headless Scope mode) delegate to this single class. This avoids code duplication and keeps the pygame viewer functional throughout all development phases.

**Major components:**
1. `simulator.py` (new) — `CASimulator`: pure simulation logic extracted from `viewer.py`; all LFO systems, advection, coverage management; `render_float()` returns `(H, W, 3) float32 [0, 1]`
2. `pipeline.py` (new) — `CAPipeline`: Scope interface; delegates to `CASimulator`; reads all user params from `kwargs` each call; returns THWC torch tensor `(1, H, W, 3)`
3. `plugin.py` (new) — `register_pipelines()` hook; single pipeline registration; the entry point Scope calls on load
4. `backend.py` (new, Phase 3) — conditional numpy/CuPy import: `try: import cupy as xp except: import numpy as xp`; all engine and simulator code uses `xp` instead of `np` — zero duplicate code paths
5. `viewer.py` (modified) — slimmed to pygame display wrapper; delegates all simulation to `CASimulator`
6. Engine files (`lenia.py`, `smoothlife.py`, `mnca.py`, `gray_scott.py`) — unchanged; pure numpy, adopt backend shim automatically

**Data flow:** Scope scheduler → `CAPipeline.__call__(**kwargs)` → `CASimulator.set_runtime_params()` → `CASimulator.step(dt)` [engine FFT on GPU] → `IridescentPipeline.render_float()` [color LUT on CPU] → `cp.asnumpy()` if GPU → `torch.from_numpy().unsqueeze(0)` → THWC tensor → Scope → Krea Realtime + LoRA

**Key design decision:** Keep `IridescentPipeline` color rendering on CPU. The LUT lookup (192KB, memory-bandwidth bound) does not benefit from GPU. Only the FFT-heavy simulation step goes on GPU. One `cp.asnumpy()` transfer per frame (~0.3ms at 512x512) is the only device boundary crossing.

### Critical Pitfalls

1. **Storing runtime params in `__init__()`** — The existing `Viewer` initializes all state (engine, flow params, LFO, preset) in `__init__()`. Lifting this pattern into the Scope plugin breaks live UI controls permanently; Scope UI sliders do nothing. Prevention: `__init__()` creates the engine and static buffers only; ALL user-controllable params are read from `kwargs` in `__call__()` on every frame.

2. **Importing pygame at module level** — `viewer.py` imports pygame at line 4. If any file in the plugin's import chain imports `viewer.py` or `controls.py`, the plugin fails to load on RunPod (no X11/SDL) with a cryptic SDL error. The plugin just won't appear in Scope's pipeline list. Prevention: `pipeline.py` imports engines directly — never imports `viewer.py` or `controls.py`.

3. **VRAM OOM with 14B model** — Wan2.1-T2V-14B requires ~28GB; RTX 5090 has 32GB; ~4GB headroom for CA + CUDA overhead + inference activation spikes. Prevention: run at 512x512 on GPU (not 1024x1024), cap CuPy memory pool to 512MB, keep color pipeline on CPU.

4. **`cupyx.scipy.ndimage.map_coordinates` wrap mode uncertainty** — Flow field advection uses `mode='wrap'`; cupyx support for this exact mode varies by version. Prevention: implement custom CuPy bilinear interpolation with `cp.mod()` for guaranteed wrap behavior — straightforward and eliminates the version risk.

5. **Wrong THWC tensor output format** — `IridescentPipeline.render()` returns `(H, W, 3) uint8`. Missing `.copy()`, missing `/ 255.0`, or missing `.unsqueeze(0)` each cause silent corruption or black frames. Prevention: exact chain — `torch.from_numpy(rgb_u8.copy()).float().div(255.0).unsqueeze(0)` — assert shape `(1, H, W, 3)` in development.

## Implications for Roadmap

Based on research, the milestone decomposes into four phases with strict sequential dependencies. Phases cannot be parallelized — each requires the previous to be complete and passing validation tests.

### Phase 1: Extract CASimulator

**Rationale:** The architectural prerequisite for all subsequent phases. Without a clean `simulator.py` that is importable without pygame, neither the Scope plugin nor the CuPy backend can be built correctly. This phase also validates that `viewer.py`'s logic can be cleanly reorganized without breaking the existing pygame interactive viewer.

**Delivers:** `simulator.py` with `CASimulator` class; `render_float()` returning `(H, W, 3) float32 [0,1]`; `viewer.py` slimmed to pygame display wrapper delegating to `CASimulator`; `python -m cellular_automata coral` continues working identically.

**Addresses:** Table stakes — headless execution, continuous frame output

**Avoids:** Pitfall 1 (init/kwargs boundary established from the start), Pitfall 2 (pygame import isolation confirmed)

**Test gate:** `python -m cellular_automata coral` produces identical visual output; headless test passes (`sys.modules['pygame'] = None` before import, `CASimulator` instantiates, `render_float(0.033)` returns `(1024, 1024, 3)` float32 in `[0,1]`).

### Phase 2: Scope Plugin Wrapper

**Rationale:** With `CASimulator` cleanly extracted, writing the Scope plugin is mechanical. The plugin contract is fully specified by the first-party `example_plugin/` in this repo. All open questions (entry point format, THWC tensor, kwargs pattern) are answered.

**Delivers:** `plugin.py` + `pipeline.py` + `pyproject.toml`; "Cellular Automata" appears in Scope pipeline selector; preset, speed, hue controls work from Scope UI; plugin installs and runs on CPU without any GPU requirement.

**Uses:** hatchling (build backend), torch (tensor conversion), `uv run daydream-scope install`

**Implements:** `CAPipeline.__call__()` Scope interface, `register_pipelines` hook, `ui_field_config()` with runtime controls panel

**Avoids:** Pitfall 5 (THWC format — exact conversion chain with assert), Pitfall 7 (warmup deferred to first `__call__()` or background thread), Pitfall 10 (rate decoupling with `time.perf_counter()`), Pitfall 13 (pyproject.toml entry point validated before first install), Pitfall 14 (LFO dt from wall-clock not pygame clock)

**Test gate:** `uv run daydream-scope install .` succeeds; Scope UI shows "Cellular Automata" pipeline; moving speed/hue sliders visibly changes output; no pygame import in plugin's import chain (verified with `sys.modules['pygame'] = None` guard).

### Phase 3: CuPy GPU Acceleration

**Rationale:** CPU performance (10-18 FPS at 1024x1024) is insufficient for real-time Scope output. GPU acceleration is required before RunPod deployment to achieve 30+ FPS. Importantly, the VRAM budget decision (512x512 on GPU) is a design constraint that must be baked in here — not retrofitted after deployment.

**Delivers:** `backend.py` numpy/CuPy conditional shim; all 4 engine files using `xp` from backend; custom bilinear wrap for advection (no cupyx.scipy dependency for this op); `IridescentPipeline` stays on CPU with one `cp.asnumpy()` transfer per frame; benchmark confirms 30+ FPS at 512x512 on RTX 5090.

**Uses:** `cupy-cuda12x>=13.3.0` (installed on RunPod only), `cupyx.scipy.ndimage` for gaussian_filter and maximum_filter, custom bilinear wrap for map_coordinates

**Avoids:** Pitfall 3 (FFT precision — visual QA on all 4 engines for 500+ steps before declaring done), Pitfall 4 (VRAM budget — 512x512, pool cap 512MB, color on CPU), Pitfall 6 (map_coordinates wrap mode — custom implementation), Pitfall 9 (scipy.ndimage.zoom not in cupyx — build static fields on CPU), Pitfall 12 (GS dead zone — explicit mask `V[V < 0.015] = 0.0`), Pitfall 15 (LUT on CPU not GPU)

**Test gate:** CPU and GPU both produce living organisms for 500+ steps; 30+ FPS benchmark at 512x512 on RTX 5090; `nvidia-smi` shows CA plugin < 200MB VRAM; CA + 14B model coexist without OOM error.

### Phase 4: RunPod Deployment

**Rationale:** Final integration test in production environment. The full CA → Scope → Krea Realtime → LoRA pipeline only validates in the real RunPod environment with the 14B model loaded and GPU actively shared.

**Delivers:** CA plugin installed on RunPod pod; `cupy-cuda12x` in container; end-to-end pipeline test; visual output validation (AI transform visible on CA frames); latency profiling.

**Uses:** `daydreamlive/scope:main` base image, croc for file transfer, SSH gateway (`-tt` flag), documented RunPod paths

**Avoids:** Pitfall 4 (load 14B model first, then CA — test VRAM headroom in correct load order), Pitfall 11 (verify CUDA version with `nvcc --version` before installing CuPy — use `cupy-cuda12x`), Pitfall 17 (audit scipy availability in container; add to deps or use scipy-free fallback)

**Test gate:** CA pipeline visible in Scope on RunPod; CA + Krea Realtime run simultaneously without OOM; AI transform is visibly applied to CA frames; acceptable latency for intended use.

### Phase Ordering Rationale

- Phase 1 must come first: `CASimulator` extraction establishes the clean import boundary that makes both Phase 2 (plugin) and Phase 3 (CuPy backend) straightforward
- Phase 2 must precede Phase 3: the Scope plugin is the integration point that validates CuPy output actually reaches Scope in the correct tensor format; testing CuPy without the plugin wrapper is incomplete
- Phase 3 must precede Phase 4: VRAM budget (512x512 on GPU) is a Phase 3 design decision; deploying a CPU-only plugin alongside 14B inference on RunPod wastes the GPU and risks CPU contention
- CA visual quality does not change across any phase — all organism behavior is frozen from the Phase 1 prototype

### Research Flags

Phases likely needing deeper research during planning:

- **Phase 3 (CuPy Port):** Verify `cupyx.scipy.ndimage.map_coordinates` mode support on the specific CuPy version available on RunPod before committing. Verify RTX 5090 (Blackwell SM_100) appears in CuPy 14.x supported GPU list — if missing, CuPy will use PTX compilation (slower first run, but functional). Benchmark FFT precision differences on all 4 engines at 500+ steps.
- **Phase 2 (Scope Plugin):** Verify whether `ui_field_config()` supports string enum/dropdown fields for preset selection. Only numeric slider examples exist in `example_plugin/`. If not supported, use integer index 0–10 mapped to `UNIFIED_ORDER`.

Phases with standard patterns (skip additional research):

- **Phase 1 (Extract CASimulator):** Pure Python refactor of existing code; no external API uncertainty; all needed patterns visible in current codebase.
- **Phase 4 (RunPod Deployment):** SSH/croc/path mechanics are battle-tested from prior sessions; documented in CLAUDE.md with exact commands.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Scope plugin API verified from first-party `example_plugin/`; CuPy package naming confirmed via PyPI; RunPod mechanics confirmed from working sessions |
| Features | HIGH | Plugin contract is fully specified from `example_plugin/`; open question on enum UI support is minor with a clean fallback (integer index) |
| Architecture | HIGH | Extraction boundary is clear from code audit; `CASimulator` interface is well-defined; data flow verified; design decisions (CPU color, 512x512 on GPU) are documented with rationale |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls (pygame import, kwargs/init boundary, THWC format, VRAM budget) are HIGH confidence from code and known constraints; CuPy precision and cupyx.ndimage wrap mode are MEDIUM — require validation on RunPod |

**Overall confidence:** HIGH for Phases 1–2 (standard patterns, first-party sources); MEDIUM for Phase 3 (CuPy GPU compat); HIGH for Phase 4 (deployment mechanics already proven).

### Gaps to Address

- **Scope UI enum/dropdown support:** Only numeric sliders confirmed. If string enum is not supported by `ui_field_config()`, implement preset as integer index 0–10 mapped to `UNIFIED_ORDER` in `CAPipeline.__call__()`. Verify experimentally during Phase 2 by checking Scope UI after first install.
- **`cupyx.scipy.ndimage.map_coordinates` `mode='wrap'` support:** Verify against CuPy 14.x docs or test on RunPod before committing to it. If unsupported, the custom bilinear wrap (documented in PITFALLS.md, Pitfall 6) is the drop-in fallback.
- **RTX 5090 (Blackwell SM_100) CuPy 14.x support:** Run `python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"` on RunPod before Phase 3. If SM_100 not listed as supported, CuPy falls back to PTX compilation — slower first run but still functional.
- **Scope `__call__()` cadence:** Does Scope call the pipeline at display frame rate (60fps) or on generation demand? Use `time.perf_counter()` tracking between calls to make the CA rate-independent regardless of cadence.
- **Krea Realtime T-dimension preference:** T=1 is valid per the pipeline contract. Krea Realtime may produce better motion output with T=4 consecutive CA frames. Test both during Phase 4 — returning T=4 requires buffering 4 frames in `CAPipeline`.

## Sources

### Primary (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/` — Scope plugin contract: THWC tensor format, kwargs pattern, `register_pipelines`, `ui_field_config`, `pyproject.toml` entry point format
- `/Users/agi/Code/daydream_scope/plugins/cellular_automata/` — Full CA codebase: engine files, viewer.py extraction boundary analysis, iridescent.py pygame-free status confirmed
- `/Users/agi/Code/daydream_scope/CLAUDE.md` — RunPod setup: SSH gateway mechanics, croc transfer protocol, LoRA paths, container image name, RTX 5090 32GB VRAM specification

### Secondary (MEDIUM confidence)
- PyPI `pip index versions cupy` — CuPy 14.0.0 confirmed as latest release as of 2026-02-17
- CuPy documentation (training knowledge) — cupyx.scipy.ndimage API compatibility list, cupy-cuda12x meta-package naming, memory pool API (`cp.get_default_memory_pool().set_limit()`)
- GPU FFT benchmark patterns — estimated speedup ratios (6-10x for FFT, 4-6x for map_coordinates) based on known cuFFT vs pocketfft performance characteristics

### Tertiary (LOW confidence — needs validation on RunPod)
- RTX 5090 (Blackwell SM_100) support in CuPy 14.x — requires verification before Phase 3
- `ui_field_config()` string enum support in Scope — requires experimental verification during Phase 2
- Krea Realtime T-dimension preference for motion quality — no documentation found; must test during Phase 4

---
*Research completed: 2026-02-17*
*Ready for roadmap: yes*
