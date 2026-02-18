---
phase: 03-extract-casimulator
plan: 01
subsystem: simulation
tags: [numpy, scipy, cellular-automata, lenia, gray-scott, smoothlife, mnca, lfo, flow-fields]

# Dependency graph
requires: []
provides:
  - "CASimulator class in simulator.py — headless simulation core with zero pygame dependency"
  - "IridescentPipeline.render_float() method returning (H,W,3) float32 [0,1]"
  - "All 4 CA engines (lenia, gray_scott, smoothlife, mnca) accessible via CASimulator"
affects: [03-02, phase-04-scope-plugin]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Headless simulator pattern: extract pygame-free core from viewer for Scope plugin use"
    - "render_float() contract: returns (H,W,3) float32 [0,1] for Scope tensor compatibility"
    - "CASimulator.set_runtime_params(**kwargs) for Scope UI parameter binding"
    - "Speed accumulator pattern: fractional sim_speed via float accumulator + engine.step() loop"

key-files:
  created:
    - "plugins/cellular_automata/simulator.py — CASimulator class, LFO systems, flow fields, all sim methods"
  modified:
    - "plugins/cellular_automata/iridescent.py — added render_float() method"

key-decisions:
  - "CASimulator does NOT auto-resize sim_size based on engine type — caller controls size (unlike viewer.py which auto-selects 512 for GS)"
  - "ENGINE_CLASSES in simulator.py only includes headless-safe engines (lenia, gray_scott, smoothlife, mnca) — not pygame-dependent ones"
  - "render_float() in IridescentPipeline is a utility wrapper; Scope plugin calls CASimulator.render_float() not IridescentPipeline.render_float() directly"
  - "_render_frame() returns raw (H,W,3) uint8 numpy array (not pygame.Surface) — key difference from viewer.py"

patterns-established:
  - "Import guard pattern: sys.modules['pygame'] = None before importing simulator — blocks any accidental pygame import"
  - "FLOW_KEYS defined in simulator.py, viewer.py imports from it"
  - "ENGINE_CLASSES defined in simulator.py (Pitfall 7 honored)"
  - "render_thickness initialized before _apply_preset (Pitfall 2 honored)"
  - "_prev_world = None initialized before _apply_preset (Pitfall 6 honored)"

requirements-completed: [SIM-01, SIM-02, SIM-04]

# Metrics
duration: 6min
completed: 2026-02-18
---

# Phase 3 Plan 01: Extract CASimulator Summary

**CASimulator headless core extracted from viewer.py with zero pygame imports — render_float(dt) returns (H,W,3) float32 for all 4 CA engines**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-18T05:54:22Z
- **Completed:** 2026-02-18T06:00:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created `simulator.py` with `CASimulator` class containing all simulation logic extracted from `viewer.py`
- Zero pygame dependency — confirmed by `sys.modules['pygame'] = None` import guard test
- `CASimulator.render_float(dt)` returns `(H,W,3) float32 [0,1]` — direct Scope tensor format
- All 4 engine types (lenia, gray_scott, smoothlife, mnca) produce non-zero living frames through CASimulator
- Added `IridescentPipeline.render_float()` as thin wrapper over `render()` for SIM-04 requirement

## Task Commits

Each task was committed atomically:

1. **Task 1: Create simulator.py with CASimulator class** - `6f47e72` (feat)
2. **Task 2: Add render_float() to IridescentPipeline** - `28428c2` (feat)

## Files Created/Modified
- `plugins/cellular_automata/simulator.py` — CASimulator class, all 4 LFO systems, flow field construction, advection, containment, stir, noise, coverage management, rendering (1311 lines)
- `plugins/cellular_automata/iridescent.py` — Added `render_float()` method (+13 lines)

## Decisions Made
- CASimulator does NOT auto-resize sim_size — caller controls grid size. The viewer's auto-resize (GS at 512, others at 1024) is a display optimization that the headless simulator does not replicate.
- ENGINE_CLASSES restricted to 4 headless-safe engines. Life, Excitable, CCA engines require separate treatment (they have pygame-independent code but weren't tested).
- `_render_frame()` returns raw numpy uint8 array instead of pygame.Surface — fundamental API difference from viewer.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed auto-resize overriding caller-specified sim_size**
- **Found during:** Task 1 verification
- **Issue:** `_apply_preset()` auto-selected target_sim based on engine type (GS=512, others=1024), which overrode the `sim_size` parameter passed to `CASimulator.__init__()`. Test `CASimulator('coral', 512)` returned `(1024, 1024, 3)`.
- **Fix:** Removed the auto-resize logic from `_apply_preset()` in simulator.py. Caller controls sim_size. Viewer.py (next plan) will keep the auto-resize behavior for display optimization.
- **Files modified:** `plugins/cellular_automata/simulator.py`
- **Verification:** `CASimulator('coral', 512).render_float(0.016)` returns `(512, 512, 3)` float32.
- **Committed in:** `6f47e72` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Essential fix — caller-controlled sim_size is the correct API contract for a headless simulator. No scope creep.

## Issues Encountered
- Plan's verify block used non-existent preset names `reactor` and `mitosis`. Actual test used `sl_gliders` (smoothlife) and `mnca_soliton` (mnca). All 4 engine types verified working.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `CASimulator` is ready for Plan 02 (viewer.py refactor to delegate to CASimulator)
- `render_float(dt)` is the exact interface the Scope plugin (Phase 4) will call
- All simulation systems verified: LFOs, flow fields, containment, coverage, rendering

## Self-Check: PASSED

All artifacts verified:
- `plugins/cellular_automata/simulator.py` — FOUND
- `plugins/cellular_automata/iridescent.py` — FOUND
- `.planning/phases/03-extract-casimulator/03-01-SUMMARY.md` — FOUND
- Commit `6f47e72` (Task 1) — FOUND
- Commit `28428c2` (Task 2) — FOUND

---
*Phase: 03-extract-casimulator*
*Completed: 2026-02-18*
