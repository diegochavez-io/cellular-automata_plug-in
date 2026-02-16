# Roadmap: DayDream Scope — Cellular Automata Plugin

## Overview

Transform the cellular automata plugin from a working prototype into a polished organic video source through six progressive phases. Fix LFO snap-back to stabilize breathing behavior, replace the 4-layer color system with unified iridescent rendering, implement safe parameter controls, clean up presets and UI, then expand to five new simulation engines (Physarum, DLA, agent-based, primordial particles, stigmergy). Every system renders through the same iridescent pipeline, producing beautiful constantly-evolving imagery for Scope's realtime video pipeline.

## Phases

- [x] **Phase 1: LFO Smoothing** - Fix oscillator snap-back, smooth sinusoidal breathing
- [ ] **Phase 2: Iridescent Color Pipeline** - Replace 4-layer system with unified oil-slick shimmer
- [ ] **Phase 3: Safe Parameter Control** - EMA smoothing prevents organism death from slider changes
- [ ] **Phase 4: Preset Cleanup & UI Simplification** - Remove bad presets, streamline control panel
- [ ] **Phase 5: Physarum & DLA Engines** - Add slime mold and diffusion aggregation simulations
- [ ] **Phase 6: Remaining Engines** - Add agent-based, primordial particles, stigmergy engines

## Phase Details

### Phase 1: LFO Smoothing
**Goal**: LFO breathing is smooth and controllable, never snaps back
**Depends on**: Nothing (first phase)
**Requirements**: LFO-01, LFO-02, LFO-03
**Success Criteria** (what must be TRUE):
  1. Organism breathes in smooth sinusoidal waves with no visible snap-back or reset
  2. LFO speed slider on control panel adjusts breathing tempo in real-time
  3. Default LFO cycle is very slow (imperceptible growth/retreat over 30+ seconds)
  4. Running any preset (Coral, Orbium, etc.) for 5+ minutes shows consistent breathing without glitches
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md — Replace state-coupled oscillator with sinusoidal LFO system + speed slider

### Phase 2: Iridescent Color Pipeline
**Goal**: Unified oil-slick shimmer replaces old 4-layer color system
**Depends on**: Phase 1
**Requirements**: CLR-01, CLR-02, CLR-03, CLR-04, CLR-05, CLR-06, CLR-07
**Success Criteria** (what must be TRUE):
  1. Organism surface displays oil-slick iridescent shimmer (thin-film interference effect visible)
  2. Entire organism slowly cycles through rainbow colors over time (global hue sweep)
  3. Different parts of organism body show different colors simultaneously (spatial prism gradient)
  4. RGB tint sliders on control panel shift overall color balance
  5. Brightness/darkness slider controls output luminance
  6. All existing engines (Lenia, Life, Excitable, Gray-Scott) render through the new pipeline
  7. Old Core/Halo/Spark/Memory layer controls are completely removed from code
  8. Performance remains under 22ms/frame at 1024x1024 resolution
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — Create iridescent cosine palette pipeline and integrate into viewer
- [ ] 02-02-PLAN.md — Add RGB tint + brightness sliders, double-click reset, remove old layer code

### Phase 3: Safe Parameter Control
**Goal**: Parameter sliders can't kill organisms through abrupt changes
**Depends on**: Phase 2
**Requirements**: SAF-01, SAF-02, SAF-03
**Success Criteria** (what must be TRUE):
  1. All engine parameter sliders (mu, sigma, kernel radius, T) use EMA smoothing with 0.3-0.5s time constant
  2. User can rapidly drag any slider from min to max without killing the organism
  3. Coral preset survives stress test: dragging mu slider from 0.12 to edges and back 5+ times
  4. Parameter values are clamped to safe bounds specific to each engine type
**Plans**: TBD

Plans:
- [ ] 03-01: TBD during planning

### Phase 4: Preset Cleanup & UI Simplification
**Goal**: Clean preset library with streamlined control panel
**Depends on**: Phase 3
**Requirements**: PRE-01, PRE-02, PRE-03, PRE-04, UI-01, UI-02, UI-03
**Success Criteria** (what must be TRUE):
  1. Primordial Soup, Aquarium, Scutim, Geminium presets are removed from preset list
  2. Coral, Cardiac Waves, Orbium, Mitosis presets remain and work correctly
  3. Each remaining preset has visually distinct appearance (different color strategies, not all rainbow)
  4. Control panel shows: preset selector, kernel radius, RGB sliders, brightness, brush size, LFO speed
  5. Old layer weight sliders (Core, Halo, Spark, Memory) are removed from UI
  6. All presets accessible via number keys 1-9
  7. Preset selector dropdown shows complete list including Lenia presets
**Plans**: TBD

Plans:
- [ ] 04-01: TBD during planning

### Phase 5: Physarum & DLA Engines
**Goal**: Two new simulation engines rendering through iridescent pipeline
**Depends on**: Phase 4
**Requirements**: ENG-01, ENG-02, ENG-06, ENG-07
**Success Criteria** (what must be TRUE):
  1. Physarum (slime mold) engine implements CAEngine interface with step(), set_params(), seed(), get_params(), apply_feedback()
  2. DLA (Diffusion Limited Aggregation) engine implements CAEngine interface
  3. Each engine has one preset accessible via number key
  4. Both engines produce organic constantly-evolving output (never static, never dies completely)
  5. Both engines work with radial containment mask (organism stays centered on screen)
  6. Both engines render through the unified iridescent color pipeline
  7. Running Physarum preset for 5+ minutes shows continuous growth/movement with no crashes
  8. Running DLA preset for 5+ minutes shows continuous growth/movement with no crashes
**Plans**: TBD

Plans:
- [ ] 05-01: TBD during planning

### Phase 6: Remaining Engines
**Goal**: Complete the simulation engine suite with three more types
**Depends on**: Phase 5
**Requirements**: ENG-03, ENG-04, ENG-05, ENG-06, ENG-07
**Success Criteria** (what must be TRUE):
  1. Agent-based simulation engine implements CAEngine interface with one preset
  2. Primordial particle system engine implements CAEngine interface with one preset
  3. Stigmergy engine implements CAEngine interface with one preset
  4. All three engines produce organic constantly-evolving output (never static, never dies)
  5. All three engines work with radial containment mask (organism stays centered)
  6. All three engines render through unified iridescent color pipeline
  7. Each engine accessible via number key in preset selector
  8. Running each preset for 5+ minutes shows continuous evolution with no crashes
  9. Performance across all engines remains under 22ms/frame at 1024x1024
**Plans**: TBD

Plans:
- [ ] 06-01: TBD during planning

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. LFO Smoothing | 1/1 | Complete | 2026-02-16 |
| 2. Iridescent Color Pipeline | 0/2 | Not started | - |
| 3. Safe Parameter Control | 0/TBD | Not started | - |
| 4. Preset Cleanup & UI Simplification | 0/TBD | Not started | - |
| 5. Physarum & DLA Engines | 0/TBD | Not started | - |
| 6. Remaining Engines | 0/TBD | Not started | - |
