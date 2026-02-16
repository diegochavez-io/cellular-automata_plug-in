# DayDream Scope — Cellular Automata Plugin

## What This Is

A DayDream Scope plugin that generates organic simulation video as input for Scope's realtime video pipeline. Built as a standalone pygame-ce viewer on macOS, featuring multiple simulation engines (Lenia, Physarum, DLA, agent-based, primordial particles, stigmergy) — each accessible as a preset. Unified iridescent color rendering makes every system look like a living, shimmering creature. Must be visually stunning and stable before deploying to RunPod.

## Core Value

The video source must produce beautiful, constantly-evolving organic imagery that feeds Scope's pipeline. Every system must look alive, iridescent, and never go black.

## Requirements

### Validated

- ✓ Lenia engine running with FFT convolution — existing
- ✓ Multiple CA engines (Lenia, Life, Excitable, Gray-Scott) — existing
- ✓ Preset system with engine switching via number keys — existing
- ✓ Interactive pygame-ce viewer with main loop — existing
- ✓ State-coupled oscillator (LFO) for mu/sigma modulation — existing
- ✓ Center noise injection for interior evolution — existing
- ✓ Radial containment mask keeping organism centered — existing
- ✓ Auto-reseed on organism death — existing
- ✓ Fractional speed system with 0.95 floor — existing
- ✓ Control panel with sliders — existing
- ✓ Mouse brush for painting on the organism — existing
- ✓ Screenshot capture — existing

### Active

- [ ] Iridescent color system — oil-slick shimmer, slow global hue sweep, spatial gradient across body (replaces 4-layer Core/Halo/Spark/Memory system)
- [ ] RGB tint controls — R, G, B sliders for overall color balance plus brightness/darkness
- [ ] Fix LFO snapping — smooth sinusoidal, never abrupt snap-back
- [ ] LFO speed slider — control breathing/undulation tempo from UI
- [ ] Safe sliders — EMA smoothing so abrupt slider moves don't kill organism
- [ ] Simplified control panel — presets, kernel radius, RGB sliders, brush size, LFO speed; remove layer weight sliders
- [ ] Cull bad presets — remove Primordial Soup, Aquarium, Scutim, Geminium
- [ ] Keep Lenia presets — Coral, Cardiac Waves, Orbium, Mitosis
- [ ] Make presets visually distinct — different color strategies, not all rainbow
- [ ] Physarum engine — slime mold simulation, one preset
- [ ] DLA engine — Diffusion Limited Aggregation, one preset
- [ ] Agent-based simulation engine — one preset
- [ ] Primordial particle system engine — one preset
- [ ] Stigmergy engine — one preset
- [ ] All new engines render through unified iridescent color pipeline
- [ ] All new engines accessible as presets via number keys

### Out of Scope

- RunPod deployment — this milestone is local polish only, deployment is next milestone
- Complex per-layer color controls — replacing with simpler RGB tint approach
- Audio reactivity — future milestone
- Mobile/touch controls — desktop only
- Multiple presets per new engine — one each for v1, expand later

## Context

- **Existing codebase**: ~15 Python files in `plugins/cellular_automata/`, well-structured with engine base class, color layers, controls, viewer, presets
- **Engine base class**: Abstract `CAEngine` with `step()`, `set_params()`, `seed()`, `get_params()`, `apply_feedback()` — new engines must implement this interface
- **Current color system**: 4 additive layers (Core, Halo, Spark, Memory) with HSV rainbow rotation — being replaced with iridescent system
- **Current LFO**: State-coupled oscillator with snap-back bug — needs fix
- **Research**: Thin-film interference via cosine model (~12ms at 1024²), EMA safe sliders (0.3-0.5s time constant), cosine palette gradients
- **Reference resources**: Nature of Code (context7), Shadertoy (context7) for simulation algorithms
- **Target**: Video source for Scope's Krea Realtime pipeline on RunPod RTX 5090
- **Performance baseline**: ~22ms/frame at 1024x1024
- **Run command**: `cd plugins && python3 -m cellular_automata coral`

## Constraints

- **Tech stack**: Python 3.10+, pygame-ce, numpy — must stay compatible with Scope plugin deployment
- **Performance**: Must maintain ~22ms/frame or better at 1024x1024 — realtime video source
- **Platform**: macOS (Darwin) for local dev, RunPod Linux for deployment
- **Visual**: Must never go black, never stop moving, speed >= 0.95
- **Engine interface**: New engines must implement `CAEngine` base class for preset system compatibility

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace 4-layer color system with iridescent pipeline | User wants oil-slick/soap-bubble shimmer, current layers too flat | — Pending |
| RGB tint instead of per-layer HSV controls | Simpler aesthetic control, less cognitive load | — Pending |
| Unified color pipeline for all engines | Consistent aesthetic across Lenia, Physarum, DLA, etc. | — Pending |
| Each new engine = one preset | Keep v1 focused, expand presets per engine in future | — Pending |
| EMA smoothing for safe sliders | Research confirms 0.3-0.5s time constant prevents organism death | — Pending |
| Cosine palette for iridescence | Pure numpy, ~0.5ms/frame, no new dependencies | — Pending |

---
*Last updated: 2026-02-16 after questioning (scope expanded to include new simulation engines)*
