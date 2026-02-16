# Requirements: DayDream Scope — Cellular Automata Plugin

**Defined:** 2026-02-16
**Core Value:** Beautiful, constantly-evolving organic imagery that feeds Scope's pipeline — every system alive, iridescent, never black.

## v1 Requirements

Requirements for local polish milestone. Each maps to roadmap phases.

### Color System

- [ ] **CLR-01**: Iridescent oil-slick shimmer renders across organism surface using thin-film interference model
- [ ] **CLR-02**: Slow global hue sweep cycles entire organism through rainbow colors over time
- [ ] **CLR-03**: Spatial gradient produces different colors across different parts of the organism body (prism effect)
- [ ] **CLR-04**: RGB tint sliders (R, G, B) control overall color balance of the output
- [ ] **CLR-05**: Brightness/darkness slider controls output luminance
- [ ] **CLR-06**: All simulation engines render through the unified iridescent color pipeline
- [ ] **CLR-07**: Old 4-layer Core/Halo/Spark/Memory system is removed

### LFO / Oscillator

- [ ] **LFO-01**: LFO breathing is smooth sinusoidal — no sudden snap-back or reset
- [ ] **LFO-02**: LFO speed slider on UI controls the breathing/undulation tempo
- [ ] **LFO-03**: Default LFO cycle is very slow — organic, imperceptible growth and retreat

### Parameter Safety

- [ ] **SAF-01**: All engine parameter sliders use EMA smoothing (0.3-0.5s time constant)
- [ ] **SAF-02**: Abruptly moving any slider does not kill the organism
- [ ] **SAF-03**: Parameter ranges are clamped to safe bounds per engine

### Control Panel

- [ ] **UI-01**: Simplified control panel with: presets, kernel radius, RGB sliders, brightness, brush size, LFO speed
- [ ] **UI-02**: Old layer weight sliders (Core, Halo, Spark, Memory) are removed from UI
- [ ] **UI-03**: Preset selector shows all available presets (Lenia + new engines)

### Presets

- [ ] **PRE-01**: Remove presets: Primordial Soup, Aquarium, Scutim, Geminium
- [ ] **PRE-02**: Keep Lenia presets: Coral, Cardiac Waves, Orbium, Mitosis
- [ ] **PRE-03**: Remaining presets are visually distinct from each other (different color strategies)
- [ ] **PRE-04**: All presets accessible via number keys

### New Engines

- [ ] **ENG-01**: Physarum engine — slime mold simulation implementing CAEngine interface, one preset
- [ ] **ENG-02**: DLA engine — Diffusion Limited Aggregation implementing CAEngine interface, one preset
- [ ] **ENG-03**: Agent-based simulation engine implementing CAEngine interface, one preset
- [ ] **ENG-04**: Primordial particle system engine implementing CAEngine interface, one preset
- [ ] **ENG-05**: Stigmergy engine implementing CAEngine interface, one preset
- [ ] **ENG-06**: Each new engine produces organic, constantly-evolving output (never static, never dies)
- [ ] **ENG-07**: Each new engine works with radial containment mask (organism stays centered)

## v2 Requirements

Deferred to future milestones. Tracked but not in current roadmap.

### Deployment

- **DEP-01**: Plugin deploys to RunPod as DayDream Scope pipeline
- **DEP-02**: Works with Krea Realtime on RTX 5090

### Extended Presets

- **EXT-01**: Multiple presets per engine (3-5 variations each)
- **EXT-02**: User-saveable preset configurations

### Audio

- **AUD-01**: Audio-reactive parameter modulation
- **AUD-02**: Beat detection driving LFO speed

### Advanced Color

- **ADV-01**: Per-engine color palettes (different iridescence per system)
- **ADV-02**: View-dependent iridescence (angle-based color shifts)

## Out of Scope

| Feature | Reason |
|---------|--------|
| RunPod deployment | Separate milestone after local polish is complete |
| Per-layer color controls | Replaced by simpler RGB tint — less cognitive load |
| Audio reactivity | Future milestone, needs audio input infrastructure |
| Mobile/touch UI | Desktop-only for local development |
| Multiple presets per new engine | One each for v1, expand later |
| GPU/shader rendering | Staying pure numpy for Scope compatibility |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLR-01 | Pending | Pending |
| CLR-02 | Pending | Pending |
| CLR-03 | Pending | Pending |
| CLR-04 | Pending | Pending |
| CLR-05 | Pending | Pending |
| CLR-06 | Pending | Pending |
| CLR-07 | Pending | Pending |
| LFO-01 | Pending | Pending |
| LFO-02 | Pending | Pending |
| LFO-03 | Pending | Pending |
| SAF-01 | Pending | Pending |
| SAF-02 | Pending | Pending |
| SAF-03 | Pending | Pending |
| UI-01 | Pending | Pending |
| UI-02 | Pending | Pending |
| UI-03 | Pending | Pending |
| PRE-01 | Pending | Pending |
| PRE-02 | Pending | Pending |
| PRE-03 | Pending | Pending |
| PRE-04 | Pending | Pending |
| ENG-01 | Pending | Pending |
| ENG-02 | Pending | Pending |
| ENG-03 | Pending | Pending |
| ENG-04 | Pending | Pending |
| ENG-05 | Pending | Pending |
| ENG-06 | Pending | Pending |
| ENG-07 | Pending | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 0
- Unmapped: 27 ⚠️

---
*Requirements defined: 2026-02-16*
*Last updated: 2026-02-16 after initial definition*
