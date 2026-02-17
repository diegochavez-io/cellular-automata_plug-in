# Phase 3: Safe Parameter Control - Research

**Researched:** 2026-02-16
**Domain:** EMA parameter smoothing, interactive parameter control, and organism survival mechanisms in real-time cellular automata
**Confidence:** MEDIUM

## Summary

This research investigates implementing safe, smooth parameter control that prevents cellular automata organisms from dying due to abrupt slider changes. The core challenges are: (1) applying exponential moving average (EMA) smoothing to parameter sliders with appropriate time constants for a "dreamy" 2-3 second transition feel, (2) implementing parameter coupling between mu and sigma to maintain Lenia organisms in viable zones, and (3) ensuring preset transitions morph smoothly rather than resetting.

The iridescent pipeline already demonstrates EMA-like smoothing for normalization values (using `smooth = max(current, smooth * 0.97)`), providing a proven pattern within the codebase. Audio VST plugins extensively use parameter smoothing to prevent "zipper noise" (audible clicks from rapid parameter changes), applying similar techniques through signal-rate interpolation and ramping over 20-100ms for glitch prevention and 1-16 bars for preset morphing.

For Lenia specifically, organisms exist in small viable subspaces of parameter space with fractal boundaries between stable and unstable regimes. The growth function G(u; μ, σ) = 2·exp(-(u-μ)²/(2σ²)) - 1 creates a "Goldilocks zone" where growth is positive near μ and negative elsewhere. Parameter coupling must maintain this balance to prevent death.

**Primary recommendation:** Use per-frame EMA smoothing with alpha calculated from desired 2-3 second time constants, implement ratio-based mu/sigma coupling that preserves the growth function's Goldilocks zone, and add optional invisible density injection when mass falls below critical thresholds.

## Standard Stack

The established libraries/tools for parameter smoothing in real-time Python applications:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 1.24+ | Array operations for EMA updates | Already in use, efficient for batch operations |
| math | stdlib | EMA alpha calculations | exp() function for time constant conversion |
| time | stdlib | Delta time for frame-rate independence | Already in use for LFO timing |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pygame-ce | 2.5+ | Event handling for sliders | Already in use, provides mouse drag events |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Per-frame EMA | Pre-computed interpolation curves | EMA adapts to variable frame rates; curves need frame prediction |
| Ratio-based coupling | Lookup table coupling | Ratios are simpler and mathematically explicit; LUTs more flexible but opaque |
| Invisible rescue | Visible "creature is sick" indicator | User explicitly wants invisibility; indicators break immersion |

**Installation:**
No additional packages required - all dependencies already in use.

## Architecture Patterns

### Recommended Project Structure
```
plugins/cellular_automata/
├── viewer.py              # Add EMA parameter wrapper class, death detection
├── controls.py            # No changes needed (double-click reset already works)
├── lenia.py              # No changes needed (set_params already exists)
├── presets.py            # No changes needed (preset switching in viewer)
└── engine_base.py        # Add get_mass() method for death detection
```

### Pattern 1: EMA Parameter Smoothing
**What:** Exponential moving average smoothing applied per-frame to target parameter values
**When to use:** Any interactive parameter that affects continuous simulations (prevents jarring transitions)
**Example:**
```python
# Source: Derived from iridescent.py smoothing pattern + audio DSP zipper-noise prevention
# https://forum.cockos.com/showthread.php?t=140619

class SmoothedParameter:
    """EMA-smoothed parameter wrapper for organic transitions."""

    def __init__(self, initial_value, time_constant=2.0):
        """
        Args:
            initial_value: Starting parameter value
            time_constant: Time (seconds) to reach 63.2% of target value
        """
        self.target = initial_value
        self.current = initial_value
        self.time_constant = time_constant
        self._alpha = None  # Computed per-frame based on dt

    def set_target(self, new_target):
        """Update target value (e.g., from slider callback)."""
        self.target = new_target

    def update(self, dt):
        """Apply EMA smoothing step (call each frame).

        Formula: alpha = 1 - exp(-dt / tau)
        Source: tau = -T_s / ln(alpha) rearranged
        https://www.dsprelated.com/showthread/comp.dsp/5126-2.php
        """
        # Compute alpha from time constant and frame delta
        self._alpha = 1.0 - math.exp(-dt / self.time_constant)
        # EMA update: current = alpha * target + (1 - alpha) * current
        self.current += self._alpha * (self.target - self.current)

    def get_value(self):
        """Get current smoothed value."""
        return self.current
```

**Key insight:** Using `alpha = 1 - exp(-dt / tau)` makes smoothing frame-rate independent. With tau=2.0 seconds and 60fps (dt≈0.0167s), alpha≈0.0083, meaning each frame blends ~0.8% of the target value. At 30fps, alpha auto-adjusts to ~1.6% to maintain the same 2-second feel.

### Pattern 2: Parameter Coupling (Mu/Sigma Dance)
**What:** Maintain viable mu/sigma ratio to keep organism in Goldilocks zone
**When to use:** Lenia engine when user adjusts mu or sigma sliders
**Example:**
```python
# Source: Lenia growth function analysis
# https://wartets.github.io/Lenia-Web/
# G(u; μ, σ) = 2·exp(-(u-μ)²/(2σ²)) - 1

class LeniaParameterCoupler:
    """Maintain viable mu/sigma relationship for organism survival."""

    def __init__(self, preset):
        # Store "healthy" ratio from working preset
        self.baseline_ratio = preset["mu"] / preset["sigma"]
        # Coupling strength: 0 = independent, 1 = locked ratio
        self.coupling_strength = 0.5  # Claude's discretion

        # Safe bounds per parameter (prevents extreme values)
        self.mu_min, self.mu_max = 0.05, 0.35
        self.sigma_min, self.sigma_max = 0.005, 0.06

    def couple_parameters(self, user_mu, user_sigma):
        """Apply coupling to maintain viable relationship.

        Returns:
            (adjusted_mu, adjusted_sigma) tuple
        """
        # Compute current ratio
        current_ratio = user_mu / max(user_sigma, 0.001)

        # Blend toward baseline ratio based on coupling strength
        target_ratio = (
            current_ratio * (1 - self.coupling_strength) +
            self.baseline_ratio * self.coupling_strength
        )

        # Apply ratio constraint while respecting user's intended "scale"
        # If user pushed mu up, increase both to maintain ratio
        scale = (user_mu + user_sigma) / 2
        adjusted_mu = scale * target_ratio / (1 + target_ratio)
        adjusted_sigma = scale / (1 + target_ratio)

        # Clamp to safe bounds
        adjusted_mu = np.clip(adjusted_mu, self.mu_min, self.mu_max)
        adjusted_sigma = np.clip(adjusted_sigma, self.sigma_min, self.sigma_max)

        return adjusted_mu, adjusted_sigma
```

**Design decision (Claude's discretion):** Using ratio-based coupling rather than lookup tables because it's mathematically explicit, preserves the Gaussian growth function's properties, and allows tuning via `coupling_strength` parameter. This "soft constraint" approach lets users still explore parameter space while gently guiding toward viable zones.

### Pattern 3: Preset Morphing
**What:** Smooth transition between preset parameters over time, no organism reset
**When to use:** When user switches presets via number keys 1-9
**Example:**
```python
# Source: Adapted from VST preset morphing patterns
# https://lame.buanzo.org/max4live_blog/unlock-new-soundscapes-crafting-dynamic-presets-with-the-dirty-vst-wrapper-preseter-morpher-20.html

class PresetMorpher:
    """Morph between presets without reseeding organism."""

    def __init__(self, morph_duration=2.5):
        """
        Args:
            morph_duration: Transition time in seconds (2-3s per user spec)
        """
        self.morph_duration = morph_duration
        self.morphing = False
        self.morph_progress = 0.0
        self.source_params = {}
        self.target_params = {}

    def start_morph(self, current_params, target_preset):
        """Begin morphing to new preset parameters."""
        self.source_params = current_params.copy()
        self.target_params = {
            k: v for k, v in target_preset.items()
            if k not in ("engine", "name", "description", "seed", "density")
        }
        self.morphing = True
        self.morph_progress = 0.0

    def update(self, dt):
        """Advance morph progress (call each frame)."""
        if not self.morphing:
            return None

        self.morph_progress += dt / self.morph_duration

        if self.morph_progress >= 1.0:
            self.morphing = False
            return self.target_params

        # Smooth interpolation (ease-in-out cubic for organic feel)
        t = self.morph_progress
        smooth_t = t * t * (3.0 - 2.0 * t)  # Smoothstep function

        # Interpolate each parameter
        morphed = {}
        for key in self.target_params:
            src = self.source_params.get(key, self.target_params[key])
            tgt = self.target_params[key]

            if isinstance(tgt, (int, float)):
                morphed[key] = src + (tgt - src) * smooth_t
            elif isinstance(tgt, list):
                # Handle kernel_peaks, kernel_widths lists
                morphed[key] = [
                    src[i] + (tgt[i] - src[i]) * smooth_t
                    for i in range(min(len(src), len(tgt)))
                ]
            else:
                morphed[key] = tgt  # Non-numeric params: snap immediately

        return morphed
```

**Audio DSP reference:** VST plugins morph presets over 1ms-60s ranges using linear or shaped interpolation. Smooth cubic interpolation (smoothstep) is preferred for organic feel vs. linear ramps which can sound/look mechanical. [Source](https://lame.buanzo.org/max4live_blog/unlock-new-soundscapes-crafting-dynamic-presets-with-the-dirty-vst-wrapper-preseter-morpher-20.html)

### Pattern 4: Death Detection and Invisible Rescue
**What:** Monitor organism mass and silently inject density if death is imminent
**When to use:** Lenia engine when mass falls below critical threshold
**Example:**
```python
# Source: Derived from existing auto-reseed logic in viewer.py lines 626-636

class SurvivalGuardian:
    """Monitors organism health and applies invisible rescue measures."""

    def __init__(self, engine):
        self.engine = engine
        self.critical_mass = 0.002  # From existing death threshold
        self.rescue_mass = 0.005    # Target mass for rescue injection
        self.rescue_cooldown = 0.0  # Prevent rescue spam

    def check_and_rescue(self, dt):
        """Monitor mass and apply invisible rescue if needed.

        Returns:
            True if rescue was applied
        """
        self.rescue_cooldown = max(0.0, self.rescue_cooldown - dt)

        mass = float(self.engine.world.mean())

        if mass < self.critical_mass and self.rescue_cooldown <= 0:
            # Invisible rescue: inject small density at center (no reseed)
            size = self.engine.size
            cy, cx = size // 2, size // 2
            radius = size // 8

            # Gentle Gaussian blob (not sharp circle)
            Y, X = np.ogrid[:size, :size]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            injection = np.exp(-(dist / radius) ** 2) * 0.05  # Very subtle

            self.engine.world = np.clip(
                self.engine.world + injection,
                0.0, 1.0
            )

            self.rescue_cooldown = 5.0  # Wait 5s before next rescue
            return True

        return False
```

**Design decision (Claude's discretion):** Using invisible density injection rather than parameter tweaking because it's more direct, doesn't interfere with EMA-smoothed parameters, and matches user's "creature is resilient" framing. The 5-second cooldown prevents rescue spam while allowing recovery from sustained hostile conditions.

### Anti-Patterns to Avoid
- **Direct slider → engine.set_params():** Causes jarring jumps. Always wrap in EMA smoothing.
- **Frame-count based smoothing:** Breaks at variable frame rates. Always use delta-time.
- **Identical EMA time constants:** Makes all parameters feel locked together. Vary slightly (2.0s-2.5s) for organic independence.
- **Reseeding on preset switch:** User explicitly wants morphing, not reset. Never reseed during preset transitions.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Smooth slider interpolation | Linear interpolation over fixed frame count | EMA with time constant | Frame-rate independent, handles variable latency, matches audio DSP patterns |
| Parameter constraint system | Hard clamps or rejection | Soft coupling with blend factors | Allows exploration while guiding toward safe zones, feels organic not locked |
| Preset transition curves | Custom easing library | Smoothstep cubic (3t²-2t³) | One-liner, standard in graphics/audio, perfect organic feel |
| Death detection | Complex heuristics on gradients/entropy | Simple mass threshold | Already proven in codebase (line 628), reliable and fast |

**Key insight:** The iridescent pipeline already demonstrates effective smoothing (`_edge_max_smooth = max(edge_max, _edge_max_smooth * 0.97)` on lines 144, 152, 168). This is equivalent to EMA with alpha≈0.03 per frame. We should use the same pattern but with time-constant based alpha for user-specified 2-3 second feel.

## Common Pitfalls

### Pitfall 1: Fixed Alpha Values Break at Variable Frame Rates
**What goes wrong:** Using `alpha = 0.05` directly gives 2-second smoothing at 60fps but 4-second smoothing at 30fps, making the app feel sluggish on slower systems.

**Why it happens:** EMA alpha is inherently frame-rate dependent. The formula `new = alpha * target + (1-alpha) * old` assumes fixed timesteps.

**How to avoid:** Calculate alpha from time constant and actual delta time: `alpha = 1 - exp(-dt / tau)`. This makes tau the frame-rate independent "feel" parameter.

**Warning signs:** Users report "laggy controls" on some systems, smoothing feels different in fullscreen vs windowed mode (due to vsync changes).

**Source:** [DSP Related Forum - Time constant calculation](https://www.dsprelated.com/showthread/comp.dsp/5126-2.php)

### Pitfall 2: Coupling Too Strong Locks Parameters
**What goes wrong:** With coupling_strength=1.0, mu and sigma become permanently locked in ratio, preventing users from exploring parameter space even in safe ranges.

**Why it happens:** 100% coupling removes all degrees of freedom. The system becomes single-parameter even though UI shows two sliders.

**How to avoid:** Use coupling_strength in 0.3-0.7 range. Low enough to allow exploration, high enough to guide away from death zones. Treat as tunable aesthetic parameter.

**Warning signs:** Users complain sliders "fight back" or "don't do what I want." Moving one slider unexpectedly changes the other's visual position.

**Source:** User-specified "delicate dance" framing suggests ~50% coupling, not full lock.

### Pitfall 3: Preset Morph Reseeds Organism
**What goes wrong:** When switching presets, calling `engine.seed()` creates visible "flash" as organism dies and reappears, breaking immersion.

**Why it happens:** Original preset application code (line 309 in viewer.py) always calls seed() because initial preset loading requires it. Morph transitions must skip this step.

**How to avoid:** Add `skip_seed` flag to preset application logic. On morph, only update parameters via `engine.set_params()`, never call `seed()`.

**Warning signs:** Users see organism "pop" or "restart" when pressing number keys. Defeats entire purpose of morphing.

**Source:** User requirement: "The creature transforms into the new preset's behavior rather than resetting."

### Pitfall 4: Rescue Spam Creates Immortal Blobs
**What goes wrong:** Injecting density every frame when mass is low creates unkillable, constantly-flickering blobs that ignore parameters.

**Why it happens:** Without cooldown, rescue applies faster than simulation can dissipate the injected matter, creating runaway positive feedback.

**How to avoid:** Add multi-second cooldown after each rescue. Let simulation stabilize before next intervention. Make injection subtle (0.02-0.05 range, not 0.5).

**Warning signs:** Organism never dies even in totally hostile parameters. Visual "pulse" every frame at center. Parameters seem disconnected from behavior.

**Source:** Existing auto-reseed logic (line 628) only triggers at `mass < 0.002`, no spam protection. Must add.

### Pitfall 5: Smoothed Parameters Drift from Slider Position
**What goes wrong:** User sets slider to 0.15, but smoothed parameter is still at 0.12 from previous value. Sliders visually disagree with actual parameters.

**Why it happens:** EMA smoothing creates lag between target (slider) and current (engine). During transition, they differ.

**How to avoid:** Accept this as inherent to smoothing. Do NOT try to move slider back to current value (creates visual jitter). Optionally show current vs. target in HUD for advanced users.

**Warning signs:** Users report "sliders lie" or "parameters wrong." Confusion about what value is actually active.

**Solution:** This is expected behavior, not a bug. The 2-3 second lag is the desired "dreamy" feel. Could add text display showing "target → current" if users need feedback.

## Code Examples

Verified patterns from codebase and research:

### EMA Time Constant to Alpha Conversion
```python
# Source: https://www.dsprelated.com/showthread/comp.dsp/5126-2.php
# Relationship: tau = -T_s / ln(alpha)
# Rearranged: alpha = exp(-T_s / tau) = exp(-dt / tau)
# For smoothing: alpha_blend = 1 - exp(-dt / tau)

import math

def ema_alpha_from_tau(dt, tau):
    """Convert time constant to EMA alpha for frame-independent smoothing.

    Args:
        dt: Frame delta time (seconds)
        tau: Time constant (seconds) - time to reach 63.2% of target

    Returns:
        Alpha blending factor for current frame
    """
    return 1.0 - math.exp(-dt / tau)

# Example usage:
# At 60fps: dt=0.0167, tau=2.0 → alpha≈0.0083
# At 30fps: dt=0.0333, tau=2.0 → alpha≈0.0165
# Both reach 63.2% of target in 2 seconds despite different frame rates
```

### Existing Smoothing Pattern from Iridescent Pipeline
```python
# Source: plugins/cellular_automata/iridescent.py lines 143-145
# Pattern: smoothed = max(current, smoothed * decay_factor)

# Line 144:
self._edge_max_smooth = max(edge_max, self._edge_max_smooth * 0.97)

# This is equivalent to EMA with alpha≈0.03 per frame (at 60fps)
# 0.97 decay = 1 - 0.03 alpha
# Time constant: tau ≈ 33 frames at 60fps = 0.55 seconds
```

### Smoothstep Cubic for Preset Morphing
```python
# Source: Standard graphics interpolation (Smoothstep function)
# https://en.wikipedia.org/wiki/Smoothstep
# Used in: GLSL, Unreal Engine, Unity, etc.

def smoothstep(t):
    """Cubic ease-in-out interpolation [0,1] → [0,1].

    Provides smooth acceleration and deceleration at boundaries.
    Superior to linear for organic transitions.
    """
    return t * t * (3.0 - 2.0 * t)

# Alternative: Quintic smoothstep (even smoother)
def smootherstep(t):
    """Quintic ease-in-out [0,1] → [0,1]."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hard parameter clamps | Soft coupling with blend factors | Audio VST ~2010s | Parameters feel organic not locked; allows exploration |
| Fixed alpha EMA | Time-constant based EMA | Audio DSP standard since 1990s | Frame-rate independence; consistent feel across systems |
| Linear interpolation | Smoothstep/cubic easing | Graphics industry ~2000s | Organic acceleration/deceleration; professional polish |
| Immediate preset switching | Morphing transitions | Audio plugins ~2015+ | Seamless preset changes; performance-friendly |

**Deprecated/outdated:**
- **Velocity-based parameter smoothing:** Adds complexity and overshoot. EMA is simpler and more predictable.
- **Frame-count based transitions:** `for i in range(120): blend = i/120` breaks at variable frame rates. Use delta-time accumulation.
- **Hard death prevention via parameter forcing:** Makes controls feel broken. Soft coupling + invisible rescue is less intrusive.

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal coupling strength for mu/sigma**
   - What we know: User wants "delicate dance," not full lock. Audio suggests 30-70% blend.
   - What's unclear: Exact value that feels best while preventing death in Coral preset stress test.
   - Recommendation: Start at 0.5, make it a tunable constant for iteration. Could expose as advanced slider later.

2. **Whether containment mask should actively inject density**
   - What we know: Current containment (line 228) uses passive decay (`world *= _containment`). Invisible rescue uses active injection.
   - What's unclear: Should containment become active (inject at center) or stay passive with separate rescue system?
   - Recommendation: Keep containment passive (simple, proven). Use separate rescue for active intervention. Cleaner separation of concerns.

3. **Rescue injection pattern (Gaussian vs uniform vs noise)**
   - What we know: Current auto-reseed uses preset's seed pattern (line 633). Rescue should be more subtle.
   - What's unclear: Best shape for invisible density injection (smooth blob vs noise vs ring).
   - Recommendation: Gaussian blob (matches existing add_blob pattern from line 550). Proven safe, predictable.

4. **Whether other engines need parameter smoothing**
   - What we know: Life, Excitable, Gray-Scott have different parameter spaces. Requirements focus on Lenia (mu, sigma, T, R).
   - What's unclear: Do other engines suffer death from abrupt parameter changes?
   - Recommendation: Implement smoothing infrastructure generically, apply to all engine parameters. Low-cost insurance against future issues.

## Sources

### Primary (HIGH confidence)
- Lenia growth function mathematics: [Lenia Web - Mathematical Foundations](https://wartets.github.io/Lenia-Web/)
- EMA time constant formula: [DSP Related Forum - Filter time constant](https://www.dsprelated.com/showthread/comp.dsp/5126-2.php)
- Codebase smoothing patterns: `plugins/cellular_automata/iridescent.py` lines 144, 152, 168
- Existing death detection: `plugins/cellular_automata/viewer.py` lines 626-636

### Secondary (MEDIUM confidence)
- VST preset morphing: [Dirty VST Wrapper Preseter Morpher](https://lame.buanzo.org/max4live_blog/unlock-new-soundscapes-crafting-dynamic-presets-with-the-dirty-vst-wrapper-preseter-morpher-20.html)
- Audio zipper noise prevention: [Cockos Forum - Parameter smoothing](https://forum.cockos.com/showthread.php?t=140619)
- Parameter space stability: [Flow-Lenia paper](https://arxiv.org/abs/2212.07906) (discusses parameter viability)

### Tertiary (LOW confidence)
- EMA Wikipedia articles: [Exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) - general overview, not implementation-specific
- WebSearch results on "slider UI best practices" - not domain-specific to CA or real-time processing

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use, zero new dependencies
- Architecture: HIGH - Patterns verified in existing codebase + audio industry standards
- Pitfalls: MEDIUM - Derived from research + logical inference, not direct CA-specific sources
- Parameter coupling specifics: LOW - Lenia papers discuss parameter spaces abstractly, not UI coupling strategies

**Research date:** 2026-02-16
**Valid until:** ~60 days (stable domain - EMA math unchanged since 1950s, pygame patterns stable)

**Implementation risk areas:**
- **Coupling strength value:** Requires iteration/tuning with real Coral preset stress testing
- **Rescue cooldown duration:** May need adjustment based on user feedback
- **Morph duration:** 2-3 seconds is user spec, but could feel too slow/fast in practice
- **Which parameters to smooth:** Research assumes mu, sigma, T, R all need smoothing; may be overkill for T and R

**Next steps for planner:**
1. Create SmoothedParameter class with time-constant based EMA
2. Add coupling logic to Lenia parameter setters
3. Implement PresetMorpher with skip_seed flag
4. Add SurvivalGuardian with cooldown
5. Stress test with Coral preset (mu 0.12→edges→0.12 drag test)
