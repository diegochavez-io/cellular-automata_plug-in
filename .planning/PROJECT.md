# DayDream Scope — Cellular Automata Plugin

## What This Is

A DayDream Scope plugin that generates organic cellular automata video as input for Scope's Krea Realtime pipeline. Four simulation engines (Lenia, SmoothLife, MNCA, Gray-Scott) produce constantly-evolving bioluminescent organisms rendered through a multi-zone iridescent color pipeline. Deploys to RunPod with GPU acceleration for realtime video generation fed into AI LoRA styling.

## Core Value

The video source must produce beautiful, constantly-evolving organic imagery that feeds Scope's pipeline. Every system must look alive, iridescent, and never go black.

## Current Milestone: v2.0 Scope Deployment

**Goal:** Deploy the CA video source as a Scope plugin on RunPod — full pipeline from cellular automata through Krea Realtime with LoRA styling.

**Target features:**
- Scope plugin wrapper (text-only pipeline, THWC tensors)
- CuPy GPU acceleration (numpy → GPU)
- RunPod deployment with RTX 5090
- Full pipeline: CA → Scope → Krea Realtime + LoRAs

## Requirements

### Validated

- ✓ Lenia engine with FFT convolution — v1.0
- ✓ Multiple CA engines (Lenia, SmoothLife, MNCA, Gray-Scott) — v1.0
- ✓ 11 curated presets with unified ordering — v1.0
- ✓ Multi-zone iridescent color pipeline (2D LUT, cosine palettes, harmonics) — v1.0
- ✓ Universal flow fields (7 types, 0.8px/step) — v1.0
- ✓ Varied blob seeds, radial containment, auto-reseed — v1.0
- ✓ Bloom, edge-weighted color, alpha depth curve — v1.0
- ✓ Performance optimized (float32, noise pool) — v1.0
- ✓ Interactive pygame-ce viewer with controls — v1.0
- ✓ LFO breathing modulation (polyrhythmic) — v1.0

### Active

- [ ] Text-only Scope pipeline class (output THWC tensors in [0,1])
- [ ] `register_pipelines` hook in plugin.py
- [ ] Map CA controls to Scope runtime `kwargs` parameters
- [ ] `ui_field_config()` for Scope UI (preset selector, speed, hue shift)
- [ ] Plugin installable via `uv run daydream-scope install`
- [ ] Port numpy → CuPy for GPU-accelerated simulation
- [ ] Port scipy.ndimage → cupyx.scipy.ndimage
- [ ] numpy/CuPy toggle based on GPU availability
- [ ] 30+ FPS at 1024x1024 on RTX 5090
- [ ] RunPod pod deployment with RTX 5090
- [ ] CA plugin running inside Scope container
- [ ] Full pipeline: CA → Scope → Krea Realtime + LoRAs
- [ ] Parameter tuning for best AI video input quality

### Out of Scope

- New CA engines (Physarum, DLA, Agent-based, etc.) — deferred from v1.0, revisit after deployment
- Safe sliders / EMA smoothing — nice-to-have after plugin works
- Simplified control panel — Scope UI replaces pygame controls
- Audio reactivity — future milestone
- Mobile/touch controls — desktop/cloud only
- Custom CUDA kernels — CuPy provides sufficient GPU acceleration

## Context

- **Existing codebase**: `plugins/cellular_automata/` — 4 engines, 11 presets, iridescent color, flow fields, all working locally
- **Scope plugin system**: Hook-based, text-only pipelines output THWC tensors in [0,1], runtime params via `kwargs` in `__call__()`
- **Target hardware**: RunPod RTX 5090, 32GB VRAM
- **Existing RunPod setup**: Pod `k48mr1qqbsotow-64411ea8`, SSH configured, 7 LoRAs already deployed
- **CuPy**: Drop-in GPU replacement for numpy, nearly identical API
- **Run command (local)**: `cd plugins && python3 -m cellular_automata coral`
- **Performance baseline (CPU)**: Lenia ~12-18 FPS, SmoothLife ~8-12, MNCA ~10-15, GS ~30-55

## Constraints

- **Scope compatibility**: Plugin must follow Scope's pipeline interface exactly (THWC, [0,1], kwargs)
- **GPU memory**: 14B Krea Realtime model + CA + CuPy must fit in 32GB VRAM
- **No pygame on server**: Plugin must run headless — no display, no pygame dependencies
- **Performance**: 30+ FPS target on GPU for smooth video source
- **Platform**: macOS local dev, Linux RunPod deployment — code must work on both

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Text-only pipeline (no video input) | CA generates frames from scratch, no camera/video needed | — Pending |
| CuPy over custom CUDA | Near-identical numpy API, minimal rewrite | — Pending |
| Runtime params via kwargs | Scope standard — preset, speed, hue controlled from UI | — Pending |
| Headless mode for server | No pygame/display on RunPod, separate render path | — Pending |
| Cosine palette for iridescence | Pure numpy/cupy, no new dependencies | ✓ Good (v1.0) |
| Flow at 0.8px/step | Sweet spot — 0.3 too slow, 2.0 explodes | ✓ Good (v1.0) |

---
*Last updated: 2026-02-17 after milestone v2.0 started*
