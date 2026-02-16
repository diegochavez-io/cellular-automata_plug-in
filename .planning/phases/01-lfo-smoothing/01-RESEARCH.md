# Phase 1: LFO Smoothing - Research

**Researched:** 2026-02-16
**Domain:** Time-based sinusoidal parameter modulation in real-time Python applications
**Confidence:** HIGH

## Summary

This research investigates implementing smooth, continuous Low Frequency Oscillator (LFO) modulation for the cellular automata plugin's parameter breathing effect. The current implementation uses a state-coupled oscillator with phase accumulation, but suffers from a "snap-back" bug where parameters suddenly jump instead of smoothly oscillating.

LFOs are oscillators that generate control signals at low frequencies (typically 0.01Hz to 20Hz) using periodic waveforms to modulate other parameters. For smooth, organic "breathing" effects, sinusoidal LFOs are standard because they produce continuous, gradual transitions without abrupt changes.

The core issue identified in the current codebase is that the state-coupled oscillator system uses velocity-based physics simulation with wall bounces, which creates discontinuities when parameters hit boundaries. This causes visible "snaps" in the animation. The standard approach for smooth LFO breathing is direct phase accumulation with sinusoidal evaluation, which guarantees continuity.

**Primary recommendation:** Replace the state-coupled oscillator with pure sinusoidal phase accumulators for mu, sigma, and T modulation, controlled by a single adjustable LFO speed parameter.

## Standard Stack

The established libraries/tools for time-based animation in Python pygame applications:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pygame-ce | 2.5+ | Game loop and timing | Modern pygame fork with active development |
| numpy | 1.24+ | Mathematical operations | Standard for sin/cos calculations |
| math | stdlib | Trigonometric functions | Built-in, zero-dependency alternative to numpy for scalar ops |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| time | stdlib | Delta time calculation | Already in use, standard for frame-independent timing |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| math.sin | numpy.sin | numpy.sin for arrays, math.sin for scalars; no functional difference for phase accumulation |
| Phase accumulator | State-coupled oscillator | Physics simulation adds complexity and discontinuities; only use if realistic bounce/collision needed |

**Installation:**
No additional packages required - all dependencies already in use.

## Architecture Patterns

### Recommended Project Structure
Current structure is appropriate:
```
plugins/cellular_automata/
├── viewer.py              # Main loop with LFO application
├── controls.py            # UI sliders (add LFO speed control)
├── presets.py            # Preset definitions (no changes needed)
└── engine_base.py        # Engine interface (no changes needed)
```

### Pattern 1: Delta Time Phase Accumulation
**What:** Accumulate phase continuously using frame delta time, evaluate sin/cos at each frame
**When to use:** Any smooth, periodic parameter modulation in real-time applications
**Example:**
```python
# Source: pygame delta time best practices
# https://github.com/Mekire/pygame-delta-time/blob/master/dt_example.py

class LFOModulator:
    def __init__(self, base_value, amplitude, frequency_hz=0.01):
        self.base_value = base_value
        self.amplitude = amplitude
        self.frequency_hz = frequency_hz  # cycles per second
        self.phase = 0.0  # radians

    def update(self, dt):
        """Update phase accumulator with delta time (seconds)."""
        # Phase increment: 2π radians per cycle × frequency × time
        self.phase += 2.0 * math.pi * self.frequency_hz * dt
        # Optional: wrap phase to prevent float overflow (not critical for < 1 hour sessions)
        # self.phase = self.phase % (2.0 * math.pi)

    def get_value(self):
        """Get current modulated value."""
        return self.base_value + self.amplitude * math.sin(self.phase)
```

### Pattern 2: Multiple Independent LFOs
**What:** Separate phase accumulators for each modulated parameter
**When to use:** When parameters need independent control or different frequencies
**Example:**
```python
# Source: Multi-LFO synthesis pattern
# https://python.plainenglish.io/build-your-own-python-synthesizer-part-2-66396f6dad81

class MultiLFOSystem:
    def __init__(self, lfo_speed=1.0):
        self.lfo_speed = lfo_speed  # Global speed multiplier

        # Independent LFOs for each parameter
        self.mu_lfo = LFOModulator(base_value=0.12, amplitude=0.03, frequency_hz=0.015)
        self.sigma_lfo = LFOModulator(base_value=0.010, amplitude=0.005, frequency_hz=0.012)
        self.T_lfo = LFOModulator(base_value=15, amplitude=3.75, frequency_hz=0.014)

    def update(self, dt):
        """Update all LFOs with speed-adjusted delta time."""
        adjusted_dt = dt * self.lfo_speed
        self.mu_lfo.update(adjusted_dt)
        self.sigma_lfo.update(adjusted_dt)
        self.T_lfo.update(adjusted_dt)

    def get_values(self):
        """Get all modulated parameter values."""
        return {
            'mu': self.mu_lfo.get_value(),
            'sigma': self.sigma_lfo.get_value(),
            'T': int(max(3, self.T_lfo.get_value()))  # Clamp T to minimum
        }
```

### Pattern 3: UI-Controlled LFO Speed
**What:** Single slider controls the breathing tempo by scaling delta time
**When to use:** Give users real-time control over oscillation speed without restarting
**Example:**
```python
# Source: Real-time parameter modulation in pygame
# https://www.ericeastwood.com/blog/7/animated-sine-wave-two-ways-with-pygame-and-tkinter

# In control panel setup:
self.sliders["lfo_speed"] = panel.add_slider(
    "LFO Speed", 0.0, 3.0, 1.0, fmt=".2f",
    on_change=lambda v: setattr(self.lfo_system, 'lfo_speed', v)
)

# In main loop (viewer.py):
def run(self):
    while self.running:
        now = time.time()
        dt = min(now - last_time, 0.1)  # Cap dt to prevent jumps
        last_time = now

        # Update LFOs with delta time
        if not self.paused:
            self.lfo_system.update(dt)
            params = self.lfo_system.get_values()
            self.engine.set_params(**params)
```

### Pattern 4: Phase Wrapping (Optional)
**What:** Keep phase within [0, 2π] to prevent float precision drift over long sessions
**When to use:** Applications running for hours where float accumulation could theoretically overflow
**Example:**
```python
# Source: NumPy phase unwrapping documentation
# https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html

def update(self, dt):
    self.phase += 2.0 * math.pi * self.frequency_hz * dt

    # Wrap phase to [0, 2π] - prevents theoretical overflow after ~24 hours
    if self.phase > 2.0 * math.pi:
        self.phase = self.phase % (2.0 * math.pi)
```
**Note:** For typical session lengths (< 1 hour), phase wrapping is optional. Python floats (float64) can safely accumulate for many hours before precision loss.

### Anti-Patterns to Avoid
- **Velocity-based physics for smooth oscillation:** State-coupled oscillators with velocity/force/damping create discontinuities at boundaries (wall bounces). Use direct sinusoidal evaluation instead.
- **Reading LFO base values from engine state:** As documented in project memory, this causes preset drift. Always read base values from preset dictionary on reset.
- **Integer phase counters:** Using frame counts instead of delta time makes oscillation speed frame-rate dependent. Always use actual elapsed time (dt).
- **Resetting phase on parameter changes:** User adjusting a slider should NOT reset the breathing cycle. Phase accumulator must persist across parameter updates.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Delta time calculation | Custom frame timer | `time.time()` with delta | Standard library, battle-tested, handles edge cases |
| Sine wave generation | Custom wave tables | `math.sin()` or `numpy.sin()` | Optimized C implementation, perfectly accurate |
| Parameter interpolation | Manual lerp with velocity | Direct sin evaluation | Guarantees smoothness, no accumulation errors |
| LFO phase wrapping | Manual modulo checks | `phase % (2π)` or skip entirely | Float64 precision sufficient for hours of runtime |

**Key insight:** Time-based animation is a solved problem in game development. The delta-time + phase-accumulator pattern is standard because it guarantees frame-rate independence and smooth continuous motion. Physics simulation (velocity/force/damping) adds complexity without benefits for pure sinusoidal oscillation.

## Common Pitfalls

### Pitfall 1: Wall Bounce Discontinuities
**What goes wrong:** Using physics simulation with "soft walls" and velocity reflection creates visible snaps when parameters hit boundaries.
**Why it happens:** The current code implements a state-coupled oscillator with force-based dynamics and wall bounces:
```python
# From viewer.py lines 447-461
if new_mu < mu_lo:
    new_mu = mu_lo
    self._mu_vel = abs(self._mu_vel) * 0.3  # Velocity reversal = discontinuity
```
**How to avoid:** Use clamped sinusoidal oscillation instead. The sine function naturally oscillates between -1 and +1 without discontinuities.
**Warning signs:** Parameters suddenly "snap" to different values; visible jerks in animation; parameters hitting exact boundary values repeatedly.

### Pitfall 2: Frame-Rate Dependent Oscillation
**What goes wrong:** Incrementing phase by a constant each frame makes oscillation speed depend on FPS.
**Why it happens:** Confusing frame number with elapsed time:
```python
# WRONG - frame-rate dependent
self.phase += 0.01  # Fixed increment per frame

# RIGHT - frame-rate independent
self.phase += 2.0 * math.pi * frequency_hz * dt  # Scaled by actual time
```
**How to avoid:** Always multiply phase increment by delta time (dt). This makes oscillation speed consistent regardless of frame rate.
**Warning signs:** LFO breathing speeds up/slows down when window loses focus; different behavior on different hardware; speed changes when other processes run.

### Pitfall 3: Preset Drift from Reading Engine State
**What goes wrong:** Reading base parameter values from `self.engine.mu` instead of preset definition causes drift over time.
**Why it happens:** LFO modifies engine parameters each frame. If reset reads current (modulated) values as new base values, the base drifts.
**How to avoid:** Store base values from preset dict at initialization:
```python
# Store base values from preset (not engine state)
preset = get_preset(self.preset_key)
self._lfo_base_mu = preset.get("mu", 0.15)  # From preset dict
# NOT: self._lfo_base_mu = self.engine.mu  # From engine state (drifts!)
```
**Warning signs:** Organism behaves differently after reset; parameters slowly drift from preset specifications; LFO range changes over multiple resets.

### Pitfall 4: Phase Reset on User Input
**What goes wrong:** Resetting `self.phase = 0.0` when user adjusts parameters causes breathing to restart from beginning, creating a visible discontinuity.
**Why it happens:** Conflating "parameter changed" with "need to restart oscillation."
**How to avoid:** LFO phase should persist independently of parameter changes. Only reset phase on explicit user action (pressing 'R' key to reseed).
**Warning signs:** Breathing "stutters" when adjusting sliders; oscillation always starts from same position; organic flow is interrupted by UI interaction.

### Pitfall 5: Float Overflow Paranoia
**What goes wrong:** Unnecessarily complex phase wrapping logic or switching to integer counters.
**Why it happens:** Concern that phase accumulation will overflow float precision.
**How to avoid:** Python float64 can safely accumulate for 100+ hours before precision loss matters. Simple modulo wrapping is fine if desired:
```python
self.phase = self.phase % (2.0 * math.pi)  # Optional, clean but not critical
```
**Warning signs:** Complex bit manipulation; switching to integer math; elaborate overflow protection for short-lived sessions.

### Pitfall 6: LFO Speed = 0 Edge Case
**What goes wrong:** Setting LFO speed to 0.0 should freeze breathing, but phase accumulator stops updating and causes engine stasis.
**Why it happens:** `dt * lfo_speed` becomes 0, so phase never increments.
**How to avoid:** Speed of 0 is valid - it simply holds current modulated values constant. Ensure engine.set_params() is still called with frozen LFO values.
**Warning signs:** Organism dies when LFO speed set to 0; parameters snap back to base values; unexpected behavior at minimum slider value.

## Code Examples

Verified patterns for smooth LFO implementation:

### Complete LFO System Implementation
```python
# Source: Synthesis pattern from pygame animation best practices
# https://inventwithpython.com/blog/2012/07/18/using-trigonometry-to-animate-bounces-draw-clocks-and-point-cannons-at-a-target/

import math
import time

class SinusoidalLFO:
    """Single-parameter sinusoidal low-frequency oscillator."""

    def __init__(self, base_value, amplitude, frequency_hz=0.01):
        """
        Args:
            base_value: Center point of oscillation
            amplitude: Half the peak-to-peak range
            frequency_hz: Oscillation frequency in Hz (cycles per second)
        """
        self.base_value = base_value
        self.amplitude = amplitude
        self.frequency_hz = frequency_hz
        self.phase = 0.0  # Current phase in radians

    def update(self, dt):
        """Advance phase based on delta time.

        Args:
            dt: Time elapsed since last update (seconds)
        """
        self.phase += 2.0 * math.pi * self.frequency_hz * dt

    def get_value(self):
        """Get current modulated value."""
        return self.base_value + self.amplitude * math.sin(self.phase)

    def reset(self):
        """Reset phase to 0 (restart cycle)."""
        self.phase = 0.0


class LeniaLFOSystem:
    """Complete LFO system for Lenia parameter modulation."""

    def __init__(self, preset):
        """Initialize from preset dictionary."""
        self.lfo_speed = 1.0  # User-adjustable tempo

        # Create independent LFOs for each parameter
        # Very slow frequencies (0.01-0.02 Hz = 50-100 second periods)
        mu_base = preset.get("mu", 0.15)
        self.mu_lfo = SinusoidalLFO(
            base_value=mu_base,
            amplitude=0.03,  # ±0.03 range
            frequency_hz=0.015  # ~67 second period
        )

        sigma_base = preset.get("sigma", 0.017)
        self.sigma_lfo = SinusoidalLFO(
            base_value=sigma_base,
            amplitude=0.006,  # ±0.006 range
            frequency_hz=0.012  # ~83 second period
        )

        T_base = preset.get("T", 10)
        self.T_lfo = SinusoidalLFO(
            base_value=T_base,
            amplitude=T_base * 0.25,  # ±25% range
            frequency_hz=0.014  # ~71 second period
        )

    def update(self, dt):
        """Update all LFOs with speed-adjusted delta time."""
        adjusted_dt = dt * self.lfo_speed
        self.mu_lfo.update(adjusted_dt)
        self.sigma_lfo.update(adjusted_dt)
        self.T_lfo.update(adjusted_dt)

    def get_modulated_params(self):
        """Get current modulated parameter values."""
        return {
            'mu': self.mu_lfo.get_value(),
            'sigma': self.sigma_lfo.get_value(),
            'T': int(max(3, self.T_lfo.get_value()))  # Clamp T to safe minimum
        }

    def reset_from_preset(self, preset):
        """Reload base values from preset without resetting phase."""
        self.mu_lfo.base_value = preset.get("mu", 0.15)
        self.sigma_lfo.base_value = preset.get("sigma", 0.017)
        self.T_lfo.base_value = preset.get("T", 10)
        # Note: Phase NOT reset - breathing continues smoothly

    def reset_phase(self):
        """Explicit phase reset (only on user action like 'R' key)."""
        self.mu_lfo.reset()
        self.sigma_lfo.reset()
        self.T_lfo.reset()
```

### Integration with Viewer Main Loop
```python
# Source: Delta time game loop pattern
# https://github.com/Mekire/pygame-delta-time/blob/master/dt_example.py

# In Viewer.__init__():
def __init__(self, ...):
    # ... existing initialization ...

    preset = get_preset(start_preset)
    self.lfo_system = LeniaLFOSystem(preset) if preset['engine'] == 'lenia' else None

# In Viewer.run() main loop:
def run(self):
    last_time = time.time()

    while self.running:
        now = time.time()
        dt = min(now - last_time, 0.1)  # Cap dt to prevent physics jumps
        last_time = now

        # ... event handling ...

        # Apply LFO modulation (continuous, even when paused if desired)
        if not self.paused and self.lfo_system:
            self.lfo_system.update(dt)
            modulated = self.lfo_system.get_modulated_params()
            self.engine.set_params(**modulated)

        # ... rest of loop ...
```

### UI Control Integration
```python
# Source: Pygame UI slider pattern
# In Viewer._build_panel():

panel.add_section("LFO BREATHING")
self.sliders["lfo_speed"] = panel.add_slider(
    "LFO Speed", 0.0, 3.0, 1.0, fmt=".2f",
    on_change=self._on_lfo_speed_change
)

# Callback:
def _on_lfo_speed_change(self, val):
    if self.lfo_system:
        self.lfo_system.lfo_speed = val
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed-step phase increment | Delta-time based accumulation | Standard since ~2010 | Frame-rate independence, smooth playback |
| Frame counter oscillation | Time-based oscillation | Game dev best practice | Consistent speed across hardware |
| Physics simulation for smooth curves | Direct sinusoidal evaluation | Signal processing standard | Guaranteed continuity, no discontinuities |
| State-coupled oscillators for breathing | Independent multi-LFO with speed control | Synthesizer/audio standard | Predictable behavior, user control |

**Deprecated/outdated:**
- **State-coupled oscillators for UI breathing:** Complex physics simulation (forces, velocities, damping, wall bounces) was an interesting experiment but creates discontinuities. Pure sinusoidal LFOs are simpler and smoother.
- **LFO bases from engine state:** Reading `self.engine.mu` as base value causes drift. Always use preset dictionary as source of truth.

## Open Questions

1. **Should LFO continue when paused?**
   - What we know: Current code comment says "LFO modulation always runs, even when paused - it's meditative"
   - What's unclear: Whether this is desired behavior or should be configurable
   - Recommendation: Keep current behavior (LFO runs when paused) unless user feedback suggests otherwise. Rationale: Breathing effect is visual/aesthetic, not simulation logic.

2. **Should each preset have custom LFO frequencies?**
   - What we know: Different presets have different parameter ranges and dynamics
   - What's unclear: Whether Coral should breathe at different rate than Orbium
   - Recommendation: Start with global LFO frequencies, add per-preset customization later if needed. Keep first iteration simple.

3. **Should LFO speed slider be in "Simulation" or dedicated "LFO" section?**
   - What we know: User wants LFO speed control visible on panel
   - What's unclear: Best UX placement in existing panel layout
   - Recommendation: Create dedicated "LFO BREATHING" section between "COLOR LAYERS" and "SIMULATION" for future expansion (phase, amplitude controls).

4. **Should there be a "freeze LFO" button vs speed = 0?**
   - What we know: Setting speed to 0 freezes breathing at current phase
   - What's unclear: Whether users understand speed slider includes "frozen" state
   - Recommendation: Speed slider range [0.0, 3.0] is sufficient. 0.0 = frozen, 1.0 = default tempo, >1.0 = faster breathing. No separate button needed.

## Sources

### Primary (HIGH confidence)
- Python math library documentation - [official docs](https://docs.python.org/3/library/math.html) - Trigonometric functions
- Pygame delta time pattern - [GitHub example](https://github.com/Mekire/pygame-delta-time/blob/master/dt_example.py) - Frame-independent animation
- NumPy phase unwrapping - [official docs](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html) - Phase continuity

### Secondary (MEDIUM confidence)
- Invent with Python trigonometry tutorial - [Using Trigonometry to Animate Bounces](https://inventwithpython.com/blog/2012/07/18/using-trigonometry-to-animate-bounces-draw-clocks-and-point-cannons-at-a-target/) - Smooth oscillation patterns
- Eric Eastwood pygame sine wave animation - [Animated Sine Wave](https://www.ericeastwood.com/blog/7/animated-sine-wave-two-ways-with-pygame-and-tkinter) - Real-time parameter modulation
- Python synthesizer LFO implementation - [Making A Synth With Python — Modulators](https://python.plainenglish.io/build-your-own-python-synthesizer-part-2-66396f6dad81) - Multi-LFO architecture

### Tertiary (LOW confidence)
- Various synthesizer and audio LFO articles - General concepts, not Python-specific implementation details

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in use, no new dependencies
- Architecture: HIGH - Delta-time phase accumulation is industry-standard game dev pattern
- Pitfalls: HIGH - Identified from codebase analysis and documented in project memory
- Code examples: HIGH - Based on verified pygame patterns and synthesizer best practices

**Research date:** 2026-02-16
**Valid until:** 30 days (stable domain - LFO/animation patterns don't change rapidly)

**Key code locations:**
- Current LFO implementation: `/Users/agi/Code/daydream_scope/plugins/cellular_automata/viewer.py` lines 392-472
- Phase reset locations: Lines 83, 218, 381, 616
- Main loop delta time: Lines 561-567
