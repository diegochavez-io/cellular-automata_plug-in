# Domain Pitfalls: Real-Time Generative Art Rendering Refactoring

**Domain:** Cellular automata visual rendering pipeline refactoring
**Researched:** 2026-02-16
**Confidence:** MEDIUM (WebSearch + domain understanding of generative systems)

## Critical Pitfalls

Mistakes that cause rewrites, total organism death, or visual discontinuities requiring rollback.

### Pitfall 1: Breaking the Organism — Parameter Interdependencies

**What goes wrong:** Lenia organisms exist in extremely narrow parameter spaces. The Coral preset lives because mu=0.12, sigma=0.010, R=20, T=15 work together as a system. Changing one control (e.g., making "feedback" slider affect sigma) breaks the delicate balance and the organism dies permanently mid-session.

**Why it happens:** Lenia parameter space is high-dimensional and sparse — organisms only exist in tiny viable niches. Research shows "those creatures are found in only a small subspace of the Lenia parameter space and are not trivial to discover." When you couple new UI controls to existing tuned parameters without understanding the phase space, you shift the organism out of its viable region.

**Consequences:**
- Organism death mid-session (goes black, never recovers)
- User loses current state, must reseed
- Controls feel "broken" because small adjustments kill everything
- Presets become unusable

**Prevention:**
1. **Never couple new controls to tuned preset parameters directly.** If adding RGB tint, implement as a post-render color transform, NOT by modifying mu/sigma/T/R.
2. **Test parameter changes with auto-reseed enabled.** If organism dies repeatedly with control adjustments, the control is unsafe.
3. **Preserve base preset parameters as const.** Store `_lfo_base_mu`, `_lfo_base_sigma` as read-only from preset dict (your code already does this for LFO — extend to all critical params).
4. **Implement parameter safety bounds.** Slider ranges should be ±10% of preset base values at most, NOT arbitrary [0, 1] ranges.
5. **Add "parameter coupling" tests.** Before refactoring, document which parameters are coupled (e.g., mu/sigma ratio determines organism type) and flag these as off-limits for new controls.

**Detection:**
- Organism repeatedly dies when specific slider is moved
- Organism flickers between alive/dead states
- Preset no longer looks like documented screenshots after control changes

**Phase warning:** Phase 2 (Color System Replacement) — if RGB tints are implemented by modifying growth function parameters, this WILL break organisms.

---

### Pitfall 2: LFO/Oscillator Phase Discontinuity — The Snap-Back Bug

**What goes wrong:** The current LFO "snaps back" suddenly — the organism grows, then abruptly jumps to a different state instead of smoothly oscillating. This is a phase discontinuity artifact where the oscillator's base value is being read from mutating state instead of a fixed reference.

**Why it happens:** Your code stores LFO bases (`_lfo_base_mu`, `_lfo_base_sigma`) but somewhere the system is reading modified engine state instead of these constants. When the LFO completes a cycle or when parameters are updated externally, it resets to the current engine value (which has drifted) rather than the original preset base. Research confirms "parameter smoothing mechanisms prevent audio artifacts when effect parameters change" — same principle applies to visual LFOs.

**Root causes:**
1. **Reading base from engine state instead of preset dict** (your CLAUDE.md warns about this: "Never read LFO bases from engine state")
2. **Phase accumulation without wrapping** — if `lfo_phase` grows unbounded, floating-point precision loss causes jumps
3. **Velocity integration without damping** — `_mu_vel` and `_sigma_vel` can accumulate numerical error
4. **Delayed feedback introducing timing mismatch** — `_mass_smooth` EMA delay creates phase lag between oscillator and feedback

**Consequences:**
- Visually jarring "reset" every 30-60 seconds
- Organism appears to "hiccup" mid-growth
- Destroys the organic breathing illusion
- User perceives system as broken

**Prevention:**
1. **ALWAYS read LFO bases from immutable preset dict.** On preset load: `self._lfo_base_mu = preset['mu']` (copy value, don't reference). NEVER: `self._lfo_base_mu = self.engine.get_params()['mu']` (this reads mutating state).
2. **Wrap phase to [0, 2π) every frame:** `self.lfo_phase = self.lfo_phase % (2 * math.pi)`
3. **Clamp velocity accumulation:** `self._mu_vel = np.clip(self._mu_vel, -max_vel, max_vel)`
4. **Use separate phase trackers for different oscillators.** If mu and sigma have different periods, don't share `lfo_phase`.
5. **Add phase continuity assertions:** After modulation, verify `abs(new_mu - old_mu) < threshold` — if delta is too large, you have a discontinuity.
6. **Implement parameter smoothing filter.** Use exponential smoothing when applying LFO: `mu_target = base + oscillation; mu_actual = 0.95 * mu_actual + 0.05 * mu_target`

**Detection:**
- Organism "jumps" to different size/density suddenly
- Parameter values snap instead of sweeping smoothly
- Oscillation period changes abruptly
- Visual "popping" or "reset" artifacts
- Scope debugging: plot `lfo_phase` and parameter values over time — discontinuities show as vertical jumps

**Phase warning:** Phase 3 (LFO Fixes) — must fix BEFORE color system replacement, or you won't know if color glitches are rendering bugs or LFO bugs.

---

### Pitfall 3: Color Space Confusion — HDR to sRGB Gamma Hell

**What goes wrong:** When replacing the 4-layer additive color system with thin-film iridescence, you're moving from simple RGB addition (linear color space) to interference-based hues (requires spectral calculation, then gamma correction for display). If you compute iridescence in linear space but blend in sRGB space (or vice versa), colors will be wrong — too dark, oversaturated, or visually flat.

**Why it happens:**
- Thin-film interference is a **physical effect** requiring spectral calculation across wavelengths (research: "calculate interference for each frequency of light separately")
- Current 4-layer system uses additive blending in [0, 1] linear space, then scales to [0, 255] for pygame
- Pygame assumes sRGB color space (gamma ~2.2) for display
- If you compute iridescence in linear RGB, then directly convert to [0, 255] without gamma correction, result is too dark
- If you compute in sRGB, then additively blend, the blending is mathematically wrong (research: "most common gamma mistake is using nonlinear color textures for shading and not applying gamma correction")

**Consequences:**
- Iridescent colors look muddy or flat instead of vibrant
- Oil-slick effect doesn't shimmer correctly
- Colors too dark (missing gamma correction) or washed out (double gamma correction)
- Presets lose visual distinctiveness

**Prevention:**
1. **Choose color space explicitly and document it.** Linear RGB for computation, sRGB for display. Add comments: `# All blending in LINEAR space until final display conversion`
2. **Apply gamma correction once, at display time:**
   ```python
   # Compute iridescence in linear RGB [0, 1]
   rgb_linear = compute_thin_film_interference(...)
   # Convert to sRGB for display
   rgb_srgb = np.where(rgb_linear <= 0.0031308,
                       rgb_linear * 12.92,
                       1.055 * rgb_linear**(1/2.4) - 0.055)
   # Scale to [0, 255]
   rgb_bytes = (rgb_srgb * 255).astype(np.uint8)
   ```
3. **Verify blending math.** Additive blending MUST happen in linear space. If using pygame blend modes, research which modes assume sRGB vs linear.
4. **Test with known reference colors.** Create a test preset with pure red (1, 0, 0 linear) → should display as (255, 0, 0) sRGB. If it doesn't, gamma is wrong.
5. **Use spectral rendering correctly.** Research warns "using only R, G, and B channels is insufficient" for thin-film — you need at least 6 wavelength samples, then convert to RGB. Don't fake it with HSV rotation.

**Detection:**
- Colors too dark overall (missing gamma correction)
- Colors oversaturated (double gamma correction)
- Iridescence doesn't match reference images (oil slick, soap bubbles)
- Additive blending produces wrong hues (e.g., red + green = yellow in linear, but gray in sRGB)

**Phase warning:** Phase 2 (Color System Replacement) — this is the #1 visual quality risk when replacing additive layers with iridescence.

---

### Pitfall 4: State Drift Bug — Preset Restoration Failure

**What goes wrong:** When user switches presets (keys 1-9) or reloads current preset (R key), the organism doesn't return to its original appearance. Instead, it looks different — slightly off-color, wrong density, or using wrong growth parameters. Over multiple preset switches, the drift compounds until presets are unrecognizable.

**Why it happens:** **Parameter coupling** where controls read/write to shared mutable state instead of immutable preset definitions. Your architecture doc warns: "LFO bases are read from preset dict, NOT from engine state, to prevent drift across preset reloads." This principle must extend to ALL parameters.

Common causes:
1. **Reading from engine state instead of preset dict:** `new_mu = self.engine.get_params()['mu']` (wrong) vs `new_mu = preset['mu']` (correct)
2. **Mutating preset dict directly:** `preset['mu'] += delta` (modifies the global PRESETS data)
3. **Slider callbacks that don't preserve preset base:** Slider sets `engine.mu = value`, but doesn't update `_preset_base_mu`
4. **Feedback loops modifying preset references:** LFO or feedback system writes back to preset dict instead of transient state

Research confirms: "State drift is when there's two components that synchronize state... if care isn't taken, the state in the observer will start to drift away from what it's supposed to be."

**Consequences:**
- Presets don't reload correctly (R key doesn't restore original)
- Preset 1 looks different after switching to Preset 2 and back
- User loses ability to explore safely (can't return to known-good state)
- Debugging becomes impossible (can't reproduce issues)

**Prevention:**
1. **Treat PRESETS dict as immutable.** Copy values on read: `base_mu = float(preset['mu'])`, never reference directly.
2. **Store preset base values separately:** On preset load, create `_preset_snapshot = preset.copy()`. All reloads read from snapshot, not PRESETS.
3. **Distinguish base vs modulated values:**
   ```python
   self._base_mu = preset['mu']  # immutable
   self._modulated_mu = self._base_mu  # can vary with LFO
   self.engine.set_params(mu=self._modulated_mu)
   ```
4. **Slider callbacks write to base, not engine:** Slider changes `self._base_mu`, then LFO applies modulation, then engine gets result. Never: slider directly calls `engine.set_params()`.
5. **Add state validation:** On preset reload, verify `engine.get_params()['mu'] == preset['mu']` before seeding. Log mismatch as warning.
6. **Implement "reset to preset" button.** Separate from reseed — restores ALL parameters to preset defaults without changing random seed.

**Detection:**
- Preset looks different after reload
- Parameter values don't match preset dict after switching
- LFO oscillates around wrong center point
- "Preset drift" — Coral preset gradually becomes Orbium over multiple reloads

**Phase warning:** Phase 1 (Refactoring Prep) — audit ALL parameter reads/writes BEFORE color system replacement.

---

## Moderate Pitfalls

Mistakes that cause delays, visual artifacts, or technical debt.

### Pitfall 5: Frame Timing Regression — Breaking the 22ms Budget

**What goes wrong:** After replacing the color system, frame time jumps from 22ms to 40ms+, making the viewer laggy and breaking real-time playback. The organism stutters or slows down.

**Why it happens:**
- **Per-frame allocations:** New iridescence system creates numpy arrays every frame instead of reusing buffers (research: "in-place operations modify existing arrays without creating new ones")
- **Python loops over arrays:** Thin-film calculation uses nested loops instead of vectorized numpy ops (research: "Python loops over NumPy arrays create temporary arrays, inflating memory usage")
- **Synchronous computation:** Computing 6+ wavelength samples for spectral iridescence in sequence instead of parallel
- **Garbage collection pressure:** High object churn triggers Python GC pauses (research: "Python's garbage collector can block execution during high object churn")

**Consequences:**
- Viewer drops below 60fps (target is ~45fps at 1024x1024 with 22ms frame budget)
- Visual stuttering destroys organic feel
- Can't deploy to Scope (requires real-time performance)
- Users perceive system as broken

**Prevention:**
1. **Pre-allocate all buffers in `__init__`:**
   ```python
   self._iridescence_buffer = np.zeros((size, size, 3), dtype=np.float32)
   self._wavelength_samples = np.zeros((size, size, 6), dtype=np.float32)
   ```
2. **Use in-place operations:** `np.add(a, b, out=result)` instead of `result = a + b`
3. **Vectorize wavelength loops:** Use broadcasting to compute all wavelengths simultaneously
4. **Profile before optimizing:** Use `cProfile` or `line_profiler` to find actual bottleneck — don't guess
5. **Keep float32 for rendering:** Don't upgrade to float64 (research: "switching to float32 cuts memory usage in half")
6. **Avoid surface recreation:** Reuse pygame.Surface, use `blit()` instead of recreating
7. **Benchmark continuously:** Add frame timing to HUD, regression-test after each change

**Detection:**
- FPS drops below 45
- Frame time exceeds 22ms (measured via `pygame.time.Clock.tick()`)
- Stuttering or hitching during rendering
- Memory usage grows over time (leak)

**Phase warning:** Phase 2 (Color System Replacement) — profile immediately after iridescence implementation, BEFORE moving to next phase.

---

### Pitfall 6: Control Range Explosion — Unsafe Slider Bounds

**What goes wrong:** New RGB tint sliders use [0.0, 2.0] range (or wider), allowing users to set values that kill the organism or produce solid black/white frames. User moves "R tint" slider to 0.0 and everything goes black permanently.

**Why it happens:**
- **Arbitrary ranges:** Sliders designed without understanding safe parameter bounds (research: "design decisions must address whether there are any values on a slider that shouldn't be accepted")
- **No biological constraints:** Lenia organism has viable parameter ranges — outside those, it dies
- **Multiplicative tints:** If RGB tint multiplies existing colors, tint=0 → black, tint=10 → white clipping
- **Lack of visual feedback:** User doesn't know they're approaching unsafe zone until organism dies

**Consequences:**
- Organism death from user exploration
- Black or white-washed frames (no recovery)
- User frustration ("controls don't work")
- Inability to create presets (safe values unknown)

**Prevention:**
1. **Use additive tints, not multiplicative:** `rgb_final = rgb_base + tint` (range: [-0.2, +0.2]) instead of `rgb_final = rgb_base * tint` (range: [0, 10])
2. **Clamp slider ranges tightly:** RGB tint ±20% max, not ±200%. Research: "setting step to 10-20% of total range prevents unnecessary precision"
3. **Implement visual safety indicators:** Slider turns yellow at 80% of safe range, red at 95%
4. **Add "reset to default" per-slider:** Double-click slider to restore preset value
5. **Soft clipping at extremes:** Use tanh() or sigmoid() to compress extreme values instead of hard clamps
6. **Document safe ranges in presets:** Add `_safe_ranges = {'R': (0.8, 1.2), 'G': (0.8, 1.2), ...}` to preset dict
7. **Test boundary conditions:** Create automated test that sweeps each slider min→max, verifies organism survives

**Detection:**
- Organism dies when slider hits min or max
- Visual output goes solid color
- Feedback from users: "slider broke everything"
- Parameter values outside [0, 1] range (for normalized params)

**Phase warning:** Phase 4 (Safe Sliders) — MUST implement before user-facing release.

---

### Pitfall 7: Preset Homogeneity — Visual Similarity Problem

**What goes wrong:** After refactoring, all presets look too similar — same general blob shape, similar colors, hard to distinguish visually. User reports "Coral and Orbium look the same now."

**Why it happens:**
- **Color system homogenization:** New RGB tint applies same color palette globally, removing per-preset hue differences (research: "maintaining a consistent color palette creates cohesive experience" — but TOO consistent = boring)
- **Lost parameter diversity:** Removing unused presets eliminates outliers (Primordial Soup, Aquarium) that anchored the visual range
- **Normalization artifacts:** If you normalize state before coloring, you compress dynamic range — all presets map to same [0, 1] output range
- **Insufficient color palette diversity:** Research shows "assessing palette similarity by sorting colors" — if all presets use similar hue ranges, they blend together

**Consequences:**
- User can't tell presets apart without reading labels
- Exploration feels pointless (everything looks the same)
- Artistic value lost (variety was a key feature)
- Preset switching becomes decorative, not functional

**Prevention:**
1. **Per-preset color palettes:** Store `hue_offset`, `saturation_boost`, `color_theme` in preset dict. Coral = warm oranges, Orbium = cool blues.
2. **Preserve visual diversity metrics:** Before culling presets, measure visual distance between all pairs (e.g., histogram comparison). Keep presets that are >0.4 different.
3. **Test with "visual regression" screenshots:** Capture screenshots of all presets before refactoring. After refactoring, verify they're still visually distinct (>30% pixel difference).
4. **Use color theory for separation:** Research confirms "color theory enables VFX artists to convey intended emotions" — assign emotional palette to each preset (aggressive, calm, chaotic).
5. **Vary iridescence parameters per-preset:** Thin-film thickness, viewing angle, refractive index should differ across presets
6. **Add texture/pattern variation:** Core density, memory decay rate, spark threshold create shape differences beyond color

**Detection:**
- Screenshots of different presets look nearly identical
- User confusion: "which preset is this?"
- Histogram similarity >0.7 between presets
- All presets cluster in same hue range (e.g., all cyan-green)

**Phase warning:** Phase 5 (Preset Curation) — validate AFTER color system is stable but BEFORE deployment.

---

### Pitfall 8: Feedback Loop Decoupling — Breaking the Organism

**What goes wrong:** The current system has "feedback" where color layer outputs feed back into the simulation (`engine.apply_feedback()`). When refactoring color system, this feedback path is deleted or accidentally decoupled, causing the organism to behave completely differently — or die.

**Why it happens:**
- **Unclear coupling:** Feedback is a subtle coupling between rendering and simulation — easy to overlook during refactoring
- **Assumption that color is output-only:** Developer assumes color system is pure visualization, doesn't realize it influences simulation
- **Missing tests:** No automated test verifies feedback loop integrity
- **Architecture docs incomplete:** Current ARCHITECTURE.md mentions feedback but doesn't explain WHY it's critical

**Consequences:**
- Organism behavior changes dramatically
- Presets no longer work as designed
- State-coupled oscillator loses feedback signal (mass tracking breaks)
- Organism becomes static instead of evolving

**Prevention:**
1. **Document feedback contract explicitly:** Add docstring to `color_layers.py:compute_feedback()` explaining how each layer influences simulation
2. **Preserve feedback interface:** New color system MUST implement `compute_feedback()` returning same shape/range as old system
3. **Add integration test:** Verify organism mass changes over time when feedback enabled vs disabled (should be different)
4. **Make feedback optional but visible:** Add UI toggle: "Feedback enabled [✓]" so user knows it exists
5. **Measure feedback influence:** Log `max(abs(feedback))` each frame — if suddenly zero, feedback is broken

**Detection:**
- Organism stops evolving (static blob)
- LFO oscillator mass signal flatlines
- Organism ignores color layer changes
- Preset behavior drastically different from baseline

**Phase warning:** Phase 2 (Color System Replacement) — preserve feedback API contract during refactoring.

---

### Pitfall 9: Thin-Film Spectral Shortcuts — Fake Iridescence

**What goes wrong:** Instead of implementing proper spectral thin-film interference (6+ wavelength samples), developer takes shortcut using HSV hue rotation or procedural noise. Result looks like "rainbow gradient" instead of oil-slick iridescence — no viewing-angle dependence, no spectral purity, no physical accuracy.

**Why it happens:**
- **Complexity avoidance:** Spectral calculation is complex (research: "calculate interference for each frequency of light separately"), so developer fakes it
- **Performance concerns:** Worried about 6x computational cost of spectral rendering
- **Misunderstanding requirements:** Thinks "iridescent = rainbow colors" without understanding physics

**Consequences:**
- Visual output looks cheap/fake (rainbow gradient instead of soap bubble)
- No viewing angle effects (iridescence should change with perspective)
- Colors too saturated or wrong hues (spectral interference has specific color progressions)
- User notices it doesn't match reference images (oil slick, CD surface)

**Prevention:**
1. **Implement true thin-film interference:** Use established algorithm (e.g., Airy's formula) with wavelength sampling
2. **Reference research:** Research warns "retrofitting causes effect to lose connection with physical mechanism" — don't fake it
3. **Sample 6+ wavelengths:** RGB is insufficient (research confirms this), use [380, 440, 490, 550, 600, 650]nm
4. **Convert spectral to RGB correctly:** Use CIE color matching functions, not simple wavelength→hue mapping
5. **Test against reference images:** Compare output to photos of soap bubbles, oil slicks — should match color progression
6. **Accept the performance cost:** If too slow, reduce resolution, don't fake the effect

**Detection:**
- Colors look like HSV rainbow instead of spectral interference
- No angle dependence (colors don't shift with view)
- Wrong color sequence (soap bubbles go magenta→green→gold, not red→yellow→green)
- User says "doesn't look like oil slick"

**Phase warning:** Phase 2 (Color System Replacement) — research thin-film physics BEFORE implementing.

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 10: Control Panel Clutter — UI Complexity Explosion

**What goes wrong:** After adding RGB tint sliders, the control panel has 15+ sliders visible simultaneously (Core, Halo, Spark, Memory layers + R, G, B tint + brightness + contrast + ...). User is overwhelmed and can't find the control they want.

**Why it happens:**
- **Additive design:** Each feature adds controls without removing old ones
- **No categorization:** All sliders flat list, no grouping
- **Missing progressive disclosure:** Advanced controls always visible

**Consequences:**
- User frustration (can't find control)
- Cognitive overload (too many options)
- Increased error rate (adjust wrong slider)

**Prevention:**
1. **Remove deprecated controls:** When replacing 4-layer system, DELETE Core/Halo/Spark/Memory sliders (don't just hide)
2. **Group related controls:** "Appearance" section (RGB tint, brightness), "Simulation" section (kernel radius, speed)
3. **Hide advanced controls:** Show only essential sliders by default, add "Advanced ▼" expander
4. **Use presets as primary interface:** Most users should pick preset, adjust 2-3 sliders max

**Detection:**
- User asks "how do I change X?" when control is visible
- More than 10 sliders visible simultaneously
- User adjusts wrong slider frequently

**Phase warning:** Phase 6 (Control Simplification) — design before implementation.

---

### Pitfall 11: Insufficient LFO Period — Strobing Effect

**What goes wrong:** LFO oscillation is too fast — organism "breathes" every 5 seconds instead of every 45 seconds. Looks frenetic instead of organic.

**Why it happens:**
- **Arbitrary period choice:** Developer sets LFO frequency without testing visual feel
- **Speed coupling:** LFO period tied to sim_speed, so slower playback = faster breathing (inverted)

**Consequences:**
- Organism looks artificial (too rhythmic)
- Visual fatigue (constant rapid change)
- Loses "slow evolution" aesthetic

**Prevention:**
1. **Use real-world time for LFO, not sim time:** `lfo_phase += dt_real * lfo_freq`, not `lfo_phase += sim_steps * lfo_freq`
2. **Test multiple periods:** 15s, 30s, 45s, 60s — pick one that feels organic
3. **Make period adjustable:** Add UI slider for testing
4. **Validate with user:** Show before/after, get feedback

**Detection:**
- Organism "breathes" visibly multiple times per minute
- Feels strobing or pulsing
- User says "too fast"

**Phase warning:** Phase 3 (LFO Fixes) — adjust after fixing discontinuity bug.

---

### Pitfall 12: Color Palette Lock-In — Can't Change Hues Later

**What goes wrong:** New color system hard-codes specific hue values (e.g., iridescence film thickness = 500nm) directly in rendering code. Later, user wants to adjust colors but can't without code changes.

**Why it happens:**
- **Magic numbers in code:** Film thickness, viewing angle, refractive index are literals, not parameters
- **No runtime controls:** No UI for adjusting color palette

**Consequences:**
- Can't tweak colors without editing code
- Presets can't have different color themes
- Limits artistic exploration

**Prevention:**
1. **Parameterize color generation:** Film thickness, angle, IOR should be variables
2. **Store in preset dict:** Each preset specifies color parameters
3. **Add runtime controls:** UI sliders for film thickness, hue offset
4. **Use color themes:** Define 5-6 named themes (cool, warm, pastel, vivid) as presets

**Detection:**
- Colors are same across all presets
- Can't adjust hue without code edit
- Artist requests color tweaks frequently

**Phase warning:** Phase 2 (Color System Replacement) — parameterize from the start.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 1: Refactoring Prep | State Drift Bug (#4) | Audit ALL parameter reads — preset dict vs engine state |
| Phase 2: Color System Replacement | Gamma Hell (#3), Fake Iridescence (#9), Feedback Decoupling (#8) | Implement spectral rendering correctly, test gamma, preserve feedback API |
| Phase 3: LFO Fixes | Phase Discontinuity (#2), Insufficient Period (#11) | Read bases from preset dict, wrap phase, test 45s period |
| Phase 4: Safe Sliders | Breaking Organism (#1), Unsafe Bounds (#6) | Never couple to mu/sigma, clamp ranges tightly, test boundary conditions |
| Phase 5: Preset Curation | Visual Similarity (#7) | Measure histogram distance, assign per-preset palettes |
| Phase 6: Control Simplification | Panel Clutter (#10) | Remove deprecated sliders, group categories |

---

## Research Confidence Assessment

| Area | Confidence | Sources |
|------|------------|---------|
| Lenia parameter sensitivity | HIGH | [Lenia paper](https://arxiv.org/abs/1812.05433) (arXiv), project CLAUDE.md |
| LFO phase discontinuity | MEDIUM | Audio LFO principles ([Native Instruments](https://blog.native-instruments.com/what-is-an-lfo/)), project code review |
| Thin-film rendering | MEDIUM | [Alan Zucconi](https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/), [Blender Projects](https://projects.blender.org/blender/blender/pulls/118477) |
| Color space / gamma | MEDIUM | [NVIDIA GPU Gems](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-24-importance-being-linear), [Khronos forums](https://community.khronos.org/t/issues-with-color-space-and-srgb-encoding-user-error-or-implementation-problem/112448) |
| State drift patterns | MEDIUM | [Terraform drift](https://www.hashicorp.com/en/blog/detecting-and-managing-drift-with-terraform), [Kubernetes drift](https://komodor.com/learn/kubernetes-configuration-drift-causes-detection-and-prevention/) |
| Pygame performance | MEDIUM | [Pygame optimization wiki](https://www.pygame.org/wiki/Optimisations), [Pygame timing docs](https://www.pygame.org/docs/ref/time.html) |
| Slider UX design | MEDIUM | [Nielsen Norman Group](https://www.nngroup.com/articles/gui-slider-controls/), [Smashing Magazine](https://www.smashingmagazine.com/2017/07/designing-perfect-slider/) |
| Preset similarity | LOW | [Color palette research](https://dl.acm.org/doi/10.1145/3450626.3459776) — need visual testing |

---

## Known Gaps

**Areas needing deeper research:**
1. **Spectral to RGB conversion best practices** — CIE color matching functions vs simple wavelength mapping (needs graphics research)
2. **Lenia parameter coupling specifics** — which parameters are independent vs coupled (needs Lenia domain expert)
3. **Real-time thin-film performance** — can we hit 22ms with 6-wavelength spectral rendering at 1024x1024? (needs profiling)
4. **Optimal LFO period for organic feel** — 45s is hypothesis, needs user testing
5. **Feedback scaling for new color system** — current feedback uses Core/Halo/Spark/Memory weights; how to map to iridescence? (needs design)

---

## Sources

### High Confidence (Official/Academic)
- [Lenia: Biology of Artificial Life (arXiv)](https://arxiv.org/abs/1812.05433)
- [The Mathematics of Thin-Film Interference - Alan Zucconi](https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/)
- [The Importance of Being Linear - NVIDIA GPU Gems](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-24-importance-being-linear)
- [NumPy Memory Optimization Guide - Spark Code Hub](https://www.sparkcodehub.com/numpy/advanced/memory-optimization/)
- [Pygame Time Module Documentation](https://www.pygame.org/docs/ref/time.html)

### Medium Confidence (Technical Community)
- [Cycles: Add thin film iridescence to Principled BSDF - Blender Projects](https://projects.blender.org/blender/blender/pulls/118477)
- [Color Space and sRGB Encoding Issues - Khronos Forums](https://community.khronos.org/t/issues-with-color-space-and-srgb-encoding-user-error-or-implementation-problem/112448)
- [Pygame Optimizations Wiki](https://www.pygame.org/wiki/Optimisations)
- [Fixing Frame Stuttering in Pygame - Mindful Chase](https://www.mindfulchase.com/explore/troubleshooting-tips/game-development-tools/fixing-frame-stuttering-and-timing-desync-in-pygame-game-loops.html)
- [NumPy Float32 Precision Issues - Python Speed](https://pythonspeed.com/articles/float64-float32-precision/)

### UX/Design Sources
- [Slider Design: Rules of Thumb - Nielsen Norman Group](https://www.nngroup.com/articles/gui-slider-controls/)
- [Designing The Perfect Slider - Smashing Magazine](https://www.smashingmagazine.com/2017/07/designing-perfect-slider/)
- [Color Theory in Real-Time VFX - Mad VFX](https://www.mad-vfx.com/blogs/color-theory-in-real-time-vfx)
- [Dynamic Color Warping for Palette Comparison - ACM Transactions](https://dl.acm.org/doi/10.1145/3450626.3459776)

### State Management
- [State Drift in Software Systems - Erik Bernhardsson](https://erikbern.com/2016/09/08/state-drift.html)
- [Detecting and Managing Drift with Terraform - HashiCorp](https://www.hashicorp.com/en/blog/detecting-and-managing-drift-with-terraform)
- [Kubernetes Configuration Drift - Komodor](https://komodor.com/learn/kubernetes-configuration-drift-causes-detection-and-prevention/)

### LFO and Oscillators
- [What is an LFO? - Native Instruments Blog](https://blog.native-instruments.com/what-is-an-lfo/)
- [Low Frequency Oscillators Guide - Lunacy Audio](https://lunacy.audio/low-frequency-oscillator-lfos/)
- [Oscillation Control in Delayed Feedback Systems - ResearchGate](https://www.researchgate.net/publication/227291364_Oscillation_Control_in_Delayed_Feedback_Systems)

---

*Research completed: 2026-02-16*
*Domain: Real-time generative art rendering pipeline refactoring*
*Researcher: GSD Project Researcher Agent*
