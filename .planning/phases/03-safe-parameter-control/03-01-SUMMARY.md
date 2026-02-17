---
phase: 03-safe-parameter-control
plan: 01
subsystem: parameter-control
completed: 2026-02-16
duration: 5 min

# Dependency graph
requires:
  - 02-02 (iridescent pipeline with LFO integration)
provides:
  - EMA-smoothed parameter infrastructure
  - Ratio-based mu/sigma coupling for organism viability
  - Invisible survival rescue mechanism
affects:
  - 03-02 (constrained sliders - will use SmoothedParameter)
  - 03-03+ (all future parameter safety features)

# Tech tracking
tech-stack:
  added:
    - Exponential Moving Average (EMA) parameter smoothing
    - Frame-rate-independent time constant system
  patterns:
    - SmoothedParameter (generic EMA wrapper)
    - LeniaParameterCoupler (ratio-based constraint)
    - SurvivalGuardian (invisible state recovery)

# File tracking
key-files:
  created:
    - plugins/cellular_automata/smoothing.py
    - plugins/test_smoothing.py
  modified:
    - plugins/cellular_automata/viewer.py
    - plugins/cellular_automata/engine_base.py

# Decisions
decisions:
  - id: ema-time-constants
    choice: Varied time constants (mu=2.0s, sigma=2.2s, T=2.5s, R=2.5s)
    rationale: Organic independence prevents lockstep parameter changes
    alternatives: [single-time-constant, user-configurable]

  - id: coupling-strength
    choice: 0.5 (50% guidance, 50% user control)
    rationale: Balances organism viability with user agency
    alternatives: [0.3-weaker, 0.7-stronger, adaptive]

  - id: rescue-mechanism
    choice: Invisible Gaussian injection (no seed/reset)
    rationale: Organic recovery without visual disruption
    alternatives: [auto-reseed, parameter-reset, disable-and-notify]

  - id: smoothing-interaction-with-lfo
    choice: LFO sets targets, SmoothedParameter applies EMA to LFO output
    rationale: Layered smoothing - LFO breathes, EMA adds dreamy lag
    alternatives: [lfo-post-ema, disable-smoothing-when-lfo-active]
---

# Phase 03 Plan 01: EMA Smoothing Infrastructure Summary

**One-liner:** Frame-rate-independent EMA parameter smoothing with mu/sigma coupling and invisible organism rescue

## What Was Built

Created the core safety infrastructure that makes slider control organic and safe:

**SmoothedParameter** - Generic EMA wrapper for any numeric parameter:
- Frame-rate-independent exponential moving average with configurable time constant
- `set_target()` called by slider callbacks, `update(dt)` called each frame
- `get_value()` returns smoothed value, `snap()` for immediate reset
- Varied time constants (mu=2.0s, sigma=2.2s, T=2.5s) create organic independence

**LeniaParameterCoupler** - Ratio-based mu/sigma coupling:
- Maintains baseline mu/sigma ratio from preset at 0.5 coupling strength
- Blends current ratio toward baseline while preserving user control
- Safe bounds: mu [0.05, 0.35], sigma [0.005, 0.06]
- Bidirectional coupling - moving either slider affects both parameters

**SurvivalGuardian** - Invisible density injection:
- Monitors organism mass each frame via `engine.get_mass()`
- Injects gentle Gaussian blob (amplitude=0.05, radius=size/8) when mass < 0.002
- 5-second cooldown between rescue attempts
- No visible restart (no seed(), no iridescent reset, no LFO phase reset)

**Viewer Integration:**
- Smoothed params created for all engine sliders in `_apply_preset`
- Slider callbacks set EMA targets instead of calling `engine.set_params` directly
- Per-frame update: LFO sets targets → EMA smooths → engine receives smoothed values
- Layered smoothing: LFO breathes parameters, EMA adds dreamy lag
- Reset (R key) snaps params immediately via `snap()`
- Replaced auto-reseed block with `SurvivalGuardian.check_and_rescue(dt)`

## How It Works

**Parameter Flow:**
1. User moves slider → callback sets `SmoothedParameter.set_target()`
2. For mu/sigma: `LeniaParameterCoupler.couple()` adjusts both targets
3. Each frame: LFO (if active) sets targets with modulated values
4. Each frame: `SmoothedParameter.update(dt)` applies EMA step
5. Each frame: Smoothed values sent to `engine.set_params()`

**EMA Formula:**
```python
alpha = 1 - exp(-dt / tau)
current += alpha * (target - current)
```

**Coupling Logic:**
```python
current_ratio = mu / sigma
blended_ratio = current_ratio * 0.5 + baseline_ratio * 0.5
adjusted_sigma = adjusted_mu / blended_ratio
# Clamp both to safe bounds
```

**Survival Rescue:**
```python
if mass < 0.002 and cooldown <= 0:
    inject_gaussian_blob(center, amplitude=0.05, radius=size/8)
    cooldown = 5.0
```

## Key Implementation Details

1. **Frame-rate independence:** All smoothing uses delta-time integration
2. **LFO + smoothing interaction:** LFO updates targets each frame, EMA smooths the LFO output
3. **Integer params:** T, R, num_states, threshold converted to int after smoothing
4. **Reset behavior:** `snap()` sets both current and target for immediate response
5. **Non-Lenia engines:** Smoothed params created for all engines, but coupler/guardian only for Lenia

## Test Coverage

Created comprehensive test suite (`test_smoothing.py`):
- SmoothedParameter drift behavior (1 frame vs 5 seconds)
- SmoothedParameter snap behavior
- LeniaParameterCoupler ratio blending
- LeniaParameterCoupler bounds enforcement
- SurvivalGuardian rescue trigger and cooldown
- Viewer integration of all components

All tests pass.

## Performance Impact

- Minimal: 3 dict lookups + 3-5 EMA updates per frame
- ~0.1ms overhead at 1024×1024 (negligible vs 21ms total frame time)
- No impact on simulation performance (parameter updates are cheap)

## Deviations from Plan

None - plan executed exactly as written.

## What This Enables

This infrastructure is the foundation for all subsequent parameter safety features in Phase 03:

- **Plan 02:** Constrained sliders (will use SmoothedParameter + bounds)
- **Plan 03:** Reset confirmation (will snap smoothed params)
- **Plan 04:** Preset morphing (will use `update_baseline()` + smooth transition)

Without this smoothing layer, aggressive slider dragging kills the organism instantly. With it, the creature survives and recovers organically.

## Next Phase Readiness

**Blockers:** None

**Concerns:** None

**Ready for:** Plan 03-02 (Constrained Sliders)

The smoothing infrastructure is fully functional and tested. All subsequent parameter control features can build on these primitives.
