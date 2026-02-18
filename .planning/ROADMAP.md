# Roadmap: DayDream Scope — Cellular Automata Plugin

## Overview

Deploy the cellular automata video source as a DayDream Scope plugin running on RunPod with GPU acceleration. Four sequential phases: extract a headless simulator core, wrap it as a Scope text-only pipeline, port the hot path to CuPy for GPU speed, then deploy alongside Krea Realtime on RunPod with LoRA styling.

Phases 1-2 (v1.0 local polish) are complete. Phases 3-6 below are the v2.0 Scope Deployment milestone. Old v1.0 phases 3-6 (parameter safety, UI cleanup, new engines) are deprioritized — Scope UI replaces pygame controls, and new engines come after deployment.

## Phases

- [x] **Phase 1: LFO Smoothing** - Fix oscillator snap-back, smooth sinusoidal breathing
- [x] **Phase 2: Iridescent Color Pipeline** - Replace 4-layer system with unified oil-slick shimmer
- [x] **Phase 3: Extract CASimulator** - Headless simulation core extracted from viewer.py
- [ ] **Phase 4: Scope Plugin Wrapper** - Text-only pipeline, Pydantic config, THWC tensor output
- [ ] **Phase 5: CuPy GPU Acceleration** - Backend shim, FFT on GPU, 30+ FPS at 512x512
- [ ] **Phase 6: RunPod Deployment** - Full pipeline: CA → Scope → Krea Realtime + LoRAs

## Phase Details

### Phase 1: LFO Smoothing
**Goal**: LFO breathing is smooth and controllable, never snaps back
**Depends on**: Nothing (first phase)
**Requirements**: LFO-01, LFO-02, LFO-03
**Status**: Complete (2026-02-16)

Plans:
- [x] 01-01-PLAN.md — Replace state-coupled oscillator with sinusoidal LFO system + speed slider

### Phase 2: Iridescent Color Pipeline
**Goal**: Unified oil-slick shimmer replaces old 4-layer color system
**Depends on**: Phase 1
**Requirements**: CLR-01, CLR-02, CLR-03, CLR-04, CLR-05, CLR-06, CLR-07
**Status**: Complete (2026-02-16)

Plans:
- [x] 02-01-PLAN.md — Create iridescent cosine palette pipeline and integrate into viewer
- [x] 02-02-PLAN.md — Add RGB tint + brightness sliders, double-click reset, remove old layer code

### Phase 3: Extract CASimulator
**Goal**: Clean `CASimulator` class in `simulator.py` that runs headlessly without pygame — the prerequisite for everything else
**Depends on**: Phase 2 (completed)
**Requirements**: SIM-01, SIM-02, SIM-03, SIM-04
**Success Criteria** (what must be TRUE):
  1. `CASimulator` class exists in `simulator.py` with `render_float(dt)` returning `(H,W,3) float32 [0,1]`
  2. `simulator.py` import chain has zero pygame imports (verified with `sys.modules['pygame'] = None` guard)
  3. `viewer.py` delegates all simulation to `CASimulator` — thin display wrapper only
  4. `python -m cellular_automata coral` produces identical visual output as before refactor
  5. `IridescentPipeline.render_float()` exists and returns correct format
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md — Create simulator.py with CASimulator class + IridescentPipeline.render_float()
- [x] 03-02-PLAN.md — Refactor viewer.py to thin wrapper + simplify __main__.py snap()

### Phase 4: Scope Plugin Wrapper
**Goal**: CA appears as "Cellular Automata" pipeline in Scope UI with live preset/speed/hue controls
**Depends on**: Phase 3
**Requirements**: PLUG-01, PLUG-02, PLUG-03, PLUG-04, PLUG-05, PLUG-06, PLUG-07, PLUG-08, PLUG-09, PLUG-10
**Success Criteria** (what must be TRUE):
  1. `uv run daydream-scope install .` succeeds with no errors
  2. "Cellular Automata" appears in Scope pipeline selector
  3. Selecting preset from dropdown switches the CA engine/preset live
  4. Moving speed slider visibly changes organism evolution rate
  5. Moving hue/brightness sliders visibly changes color output
  6. Reseed toggle triggers new organism seed
  7. Output is THWC `(1, H, W, 3) float32 [0,1]` — verified by Scope accepting frames
  8. No pygame import anywhere in the installed plugin package
  9. Plugin loads instantly (warmup deferred, no init blocking)
  10. LFO breathing works at correct tempo (wall-clock dt, not fixed dt)
**Plans**: TBD

Plans:
- [ ] 04-01: TBD during planning

### Phase 5: CuPy GPU Acceleration
**Goal**: 30+ FPS at 512x512 on RTX 5090 using CuPy backend
**Depends on**: Phase 4
**Requirements**: GPU-01, GPU-02, GPU-03, GPU-04, GPU-05, GPU-06, GPU-07, GPU-08
**Success Criteria** (what must be TRUE):
  1. `backend.py` auto-detects CuPy on GPU machines, falls back to numpy on CPU
  2. All 4 engines produce living organisms for 500+ steps on both CPU and GPU
  3. MNCA threshold rules and GS dead zone (V<0.015) work correctly on GPU
  4. Benchmark confirms 30+ FPS at 512x512 on RTX 5090
  5. `nvidia-smi` shows CA plugin using < 200MB VRAM
  6. CuPy memory pool capped at 512MB
  7. IridescentPipeline runs on CPU with one `cp.asnumpy()` transfer per frame
  8. Plugin still works correctly on CPU-only machines (local dev)
**Plans**: TBD

Plans:
- [ ] 05-01: TBD during planning

### Phase 6: RunPod Deployment
**Goal**: Full end-to-end pipeline running on RunPod — CA → Scope → Krea Realtime + LoRAs
**Depends on**: Phase 5
**Requirements**: DEPL-01, DEPL-02, DEPL-03, DEPL-04, DEPL-05
**Success Criteria** (what must be TRUE):
  1. CA plugin installed and visible in Scope on RunPod pod
  2. `cupy-cuda12x` installed and verified compatible with RTX 5090 CUDA version
  3. CA + 14B Krea Realtime model run simultaneously without OOM error
  4. AI transform from Krea Realtime + LoRAs visibly applied to CA frames
  5. Acceptable latency — no dropped frames, smooth video output
  6. Preset switching and slider controls work from Scope UI on RunPod
**Plans**: TBD

Plans:
- [ ] 06-01: TBD during planning

## Phase Ordering Rationale

- **Phase 3 first**: `CASimulator` extraction creates the clean import boundary that makes the Scope plugin (Phase 4) and CuPy backend (Phase 5) straightforward. Without it, viewer.py is a pygame-dependent monolith that can't be imported headlessly.
- **Phase 4 before 5**: The Scope plugin validates that frames actually reach Scope in the correct format. Testing CuPy output without the plugin wrapper is incomplete — you need the full pipeline to verify tensor format, kwargs flow, and UI controls.
- **Phase 5 before 6**: GPU acceleration must be designed and tested before deployment. VRAM budget (512x512, 512MB pool cap) is a Phase 5 design decision that prevents OOM alongside the 14B model.
- **Phase 6 last**: Deployment is pure integration — install, configure, verify. All code must be working before this phase.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. LFO Smoothing | 1/1 | Complete | 2026-02-16 |
| 2. Iridescent Color Pipeline | 2/2 | Complete | 2026-02-16 |
| 3. Extract CASimulator | 2/2 | Complete | 2026-02-18 |
| 4. Scope Plugin Wrapper | 0/TBD | Not started | - |
| 5. CuPy GPU Acceleration | 0/TBD | Not started | - |
| 6. RunPod Deployment | 0/TBD | Not started | - |

---
*Created: 2026-02-16*
*Updated: 2026-02-18 — Phase 3 complete (viewer.py thin wrapper, snap() simplified, visual parity confirmed)*
