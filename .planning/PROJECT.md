# DayDream Scope — Cellular Automata Plugin

## What This Is

A DayDream Scope plugin that generates cellular automata video as input for Scope's realtime video pipeline. Currently running as a standalone pygame-ce viewer on macOS, being polished for visual beauty and control simplicity before deploying to RunPod as a Scope pipeline. The organism should look like a living, iridescent creature floating in frame — always moving, always evolving.

## Core Value

The organism must be visually stunning and never stop moving. If everything else fails, the video source must produce beautiful, constantly-evolving imagery that feeds Scope's pipeline.

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

- [ ] Iridescent color system — oil-slick shimmer across surface, slow global hue sweep, spatial gradient across organism body (replaces current 4-layer Core/Halo/Spark/Memory system)
- [ ] RGB tint controls — R, G, B sliders for overall color balance plus brightness/darkness control
- [ ] Cull presets — remove Primordial Soup, Aquarium, Scutim, Geminium; keep Coral, Cardiac Waves, Orbium, Mitosis
- [ ] Fix LFO snapping — organism grows then jumps back suddenly; must be smooth sinusoidal, never abrupt
- [ ] Safe sliders — moving controls should not kill the on-screen organism if adjusted abruptly
- [ ] Simplified control panel — presets, kernel radius, RGB color sliders, brush size; hide Core/Halo/Spark/Memory layer controls
- [ ] Make remaining presets visually distinct — current presets look too similar to each other
- [ ] Slower LFO overall — the breathing/undulation cycle should be very slow, organic growth

### Out of Scope

- RunPod deployment — this milestone is local polish only, deployment is next
- New CA engines — focus on Lenia engine and existing presets
- Complex per-layer color controls — replacing with simpler RGB tint approach
- Audio reactivity — not in scope for this milestone
- Mobile/touch controls — desktop only

## Context

- **Existing codebase**: ~15 Python files in `plugins/cellular_automata/`, well-structured with engine base class, color layers, controls, viewer, presets
- **Current color system**: 4 additive layers (Core, Halo, Spark, Memory) with HSV rainbow rotation — being replaced with iridescent system
- **Current LFO**: State-coupled oscillator where mu drifts right, sigma drifts left, mass provides delayed feedback — has a snapping/reset bug
- **Target**: Video source for Scope's Krea Realtime pipeline (Wan2.1-T2V-14B) on RunPod RTX 5090
- **Performance baseline**: ~22ms/frame at 1024x1024 with float32 + pre-allocated buffers
- **Run command**: `cd plugins && python3 -m cellular_automata coral`

## Constraints

- **Tech stack**: Python 3.10+, pygame-ce, numpy — must stay compatible with eventual Scope plugin deployment
- **Performance**: Must maintain ~22ms/frame or better at 1024x1024 — this is a realtime video source
- **Platform**: macOS (Darwin) for local development, RunPod Linux for deployment
- **Visual**: Organism must never go black, never stop moving, speed >= 0.95

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace 4-layer color system with iridescent shader | User wants oil-slick/soap-bubble look, current layers don't produce it | — Pending |
| RGB tint instead of per-layer HSV controls | Simpler aesthetic control, less cognitive load | — Pending |
| Remove 4 presets, keep 4 | Primordial Soup, Aquarium, Scutim, Geminium look too similar | — Pending |
| Focus on Lenia engine only | Other engines (Life, Gray-Scott, Excitable) not needed for video source | — Pending |

---
*Last updated: 2026-02-16 after initialization*
