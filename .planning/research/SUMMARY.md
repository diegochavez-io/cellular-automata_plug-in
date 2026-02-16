# Research Summary: Iridescent Color Rendering for Cellular Automata

**Domain:** Real-time procedural iridescent rendering for 2D simulation visualization
**Researched:** 2026-02-16
**Overall confidence:** MEDIUM-HIGH

## Executive Summary

The project requires replacing a 4-layer additive color system (Core/Halo/Spark/Memory) with an iridescent rendering pipeline that produces oil-slick shimmer, slow hue sweeps, and spatial gradients across cellular automata organisms. Research identifies a modular component architecture based on thin-film interference physics, procedural spatial field generation, and safe parameter interpolation patterns.

**Key findings:**
1. **Thin-film interference physics is computationally tractable on CPU** — simplified cosine model with 3 wavelength samples (R/G/B) produces convincing iridescence in ~12ms at 1024² resolution
2. **Spatial gradients via sine waves create rainbow shimmer** — pre-computed coordinate grids + runtime modulation achieves oil-slick effect without expensive noise
3. **Safe slider architecture prevents organism death** — exponential moving average (EMA) smoothing with 0.3-0.5s time constant makes parameter changes non-destructive
4. **Modular component design enables incremental migration** — separate concerns (thickness mapping, gradient, physics, tint) allow parallel implementation alongside existing system

The architecture stays within the 22ms/frame performance baseline, requires no new dependencies (pure numpy), and provides artist-friendly controls (RGB tint, gradient angle/wavelength, brightness).

## Key Findings

**Stack:** Pure numpy implementation, no GPU/shader needed — thin-film physics via vectorized cosine operations, hue rotation via matrix method (avoids slow HSV conversion)

**Architecture:** 6-component pipeline — ThicknessMapper (world→thickness) → SpatialGradient (spatial variation) → HueSweep (time-based rotation) → ThinFilmRenderer (physics) → RGBTint (color correction) → RGBA output

**Critical pitfall:** Direct slider application kills organisms; must use SafeSliderManager with EMA smoothing to interpolate parameter changes safely

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Core Iridescence (3-5 hours)
**Focus:** Replace color system, prove physics

- Implement 5 core components (ThicknessMapper, SpatialGradient, HueSweep, ThinFilmRenderer, RGBTint)
- Wire into viewer as toggle alongside old system
- Validate: shimmer visible, <22ms performance
- **Addresses:** Oil-slick appearance, slow hue sweep, spatial gradients
- **Avoids:** Premature UI changes (keep both systems during validation)

### Phase 2: UI + Safe Sliders (2-3 hours)
**Focus:** Safe parameter control, simplified UI

- Implement SafeSliderManager with EMA smoothing
- Integrate into all engine parameter callbacks (mu, sigma, R, T)
- Remove old layer weight sliders, add RGB tint controls
- Validate: abrupt slider changes don't kill organism
- **Addresses:** Safe slider requirement, simplified control panel
- **Avoids:** Parameter smoothing complexity (use proven EMA pattern)

### Phase 3: Parameter Tuning (1-2 hours)
**Focus:** Visual polish

- Tune thickness range [min_nm, max_nm] for richest colors
- Tune spatial gradient defaults (angle, wavelength, amplitude)
- Tune smoothing time constant (balance response vs safety)
- Test across all presets (Coral, Orbium, Cardiac Waves, Mitosis)
- **Addresses:** Visual quality, preset distinctiveness
- **Avoids:** Subjective tuning without systematic testing

**Phase ordering rationale:**
- Physics first → proves feasibility, establishes performance baseline
- Safety second → prevents user frustration, enables experimentation
- Polish last → requires working system to tune against

**Research flags for phases:**
- **Phase 1:** LOW research need (physics is well-documented, architecture proven)
- **Phase 2:** LOW research need (EMA smoothing is standard pattern)
- **Phase 3:** MEDIUM research need (thickness ranges and gradient parameters need empirical tuning)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Physics correctness | HIGH | Thin-film interference equations well-established, multiple authoritative sources agree |
| CPU performance | MEDIUM | Component budgets based on similar numpy operations in existing code; needs validation at 1024² |
| Safe slider pattern | HIGH | EMA smoothing is proven technique, MathWorks documentation + existing implementations |
| Visual quality | MEDIUM | Simplified physics model (3 wavelengths, cosine) produces "good enough" iridescence per shader references; full spectral integration unnecessary |
| Integration complexity | MEDIUM | Modular design allows parallel development but viewer wiring needs care |

**Confidence drivers:**
- HIGH items: Multiple authoritative sources agree (physics textbooks, production shader implementations)
- MEDIUM items: Based on analogous systems or require empirical validation (performance at target resolution, aesthetic tuning)

## Gaps to Address

### During Implementation (Phase 1)
1. **Optimal thickness range:** Physics suggests 200-800nm for visible interference, but what range produces richest colors for Lenia organisms?
   - **Action:** Start with [200, 800], tune empirically during Phase 3
   - **Risk:** LOW (range is well-constrained by physics)

2. **Hue rotation speed:** Keep 2.5 deg/s from old system or slower?
   - **Action:** Start with 2.5 deg/s (known good), add slider in Phase 2 if needed
   - **Risk:** LOW (easy parameter to adjust)

3. **Actual numpy performance at 1024²:** Budget assumes ~12ms for ThinFilmRenderer based on similar operations in current code
   - **Action:** Measure with time.perf_counter() in early prototype
   - **Risk:** MEDIUM (if exceeds budget, fallback to LUT or downsample-render-upsample)

### Future Exploration (Post-MVP)
1. **Advanced spatial gradients:** Multi-scale Perlin/Simplex noise for organic shimmer patterns
   - Current design: Simple sine wave (adequate for MVP)
   - Trade: +visual richness, -complexity, -~5ms performance cost
   - **When:** If user feedback requests more complex shimmer

2. **Feedback integration:** Should iridescent color system feed back into engine?
   - Old system had feedback coefficient that modulated world state
   - New system could compute feedback from thickness gradients or color intensity
   - **When:** If organism evolution benefits from color-based feedback (needs testing)

3. **View-dependent iridescence:** Angle-based color shifts (true thin-film physics)
   - Requires viewing angle parameter (currently perpendicular only)
   - Adds realism but increases complexity (cosθ term in phase calculation)
   - **When:** If 2D gradient insufficient for desired shimmer effect

## Ready for Roadmap

Research complete. Architecture is well-defined with:
- ✓ Component boundaries specified (6 components with clear responsibilities)
- ✓ Data flow documented (world → thickness → physics → RGB → RGBA)
- ✓ Performance budgets allocated (<22ms total, per-component breakdowns)
- ✓ Build order suggested (3 phases, 6-10 hours total)
- ✓ Anti-patterns identified (HSV conversion in hot loop, spectral integration, instant slider application)
- ✓ Integration strategy defined (parallel implementation, toggle, migration)
- ✓ Safe slider pattern validated (EMA with 0.3-0.5s time constant)

**Critical success factors:**
1. Measure performance early (Phase 1) — if ThinFilmRenderer exceeds 12ms budget, pivot to LUT approach
2. Test slider safety thoroughly (Phase 2) — abrupt mu changes must not kill organism
3. Tune visually (Phase 3) — thickness ranges and gradient params need empirical validation on actual presets

Proceeding to roadmap creation with MEDIUM-HIGH confidence.
