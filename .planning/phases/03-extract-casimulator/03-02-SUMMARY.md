---
phase: 03-extract-casimulator
plan: 02
subsystem: simulation
tags: [pygame, cellular-automata, refactor, simulator, viewer]

# Dependency graph
requires:
  - phase: 03-01
    provides: "CASimulator class in simulator.py with all simulation logic, render_float()"
provides:
  - "viewer.py as thin pygame display wrapper delegating all simulation to CASimulator"
  - "Simplified snap() in __main__.py using CASimulator (~40 lines, was ~140)"
  - "Phase 3 complete: CASimulator is now the single authoritative simulation path"
affects: [phase-04-scope-plugin]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Thin wrapper pattern: viewer.py owns display/UI only, zero simulation logic"
    - "self.simulator.step(dt) returns numpy RGB array; viewer converts to pygame Surface"
    - "All panel callbacks go through self.simulator.* — no direct engine access from viewer"
    - "snap() in 40 lines: CASimulator(preset, size) + step loop + save"

key-files:
  created: []
  modified:
    - "plugins/cellular_automata/viewer.py — 474 lines (down from 1765); pure display wrapper"
    - "plugins/cellular_automata/__main__.py — 128 lines (down from 256); snap() uses CASimulator"

key-decisions:
  - "viewer.py keeps ENGINE_LABELS dict (display-only label mapping) — not moved to simulator.py"
  - "Paused state re-renders via self.simulator._render_frame(dt) without advancing simulation state"
  - "_on_preset_select rebuilds panel after apply_preset (engine type may change slider defs)"

patterns-established:
  - "simulator.step(dt) is the single call for advance + render — viewer never touches engine directly in run loop"
  - "FLOW_SLIDER_DEFS kept in viewer.py (panel layout config, not sim logic)"

requirements-completed: [SIM-03]

# Metrics
duration: 11min
completed: 2026-02-18
---

# Phase 3 Plan 02: Refactor Viewer to Delegate to CASimulator Summary

**viewer.py reduced from 1765 to 474 lines — all simulation logic removed, delegates to self.simulator = CASimulator; snap() from 140 to 40 lines; visual output confirmed identical by user**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-18T06:03:36Z
- **Completed:** 2026-02-18T06:14:59Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- viewer.py is now a thin pygame display wrapper: 474 lines (down from 1765), zero simulation methods, 74 `self.simulator` references
- All panel callbacks go through `self.simulator.*` — speed, thickness, hue, brightness, flow fields, presets all routed through CASimulator
- snap() in __main__.py reduced from ~140 to ~40 lines using `CASimulator(preset, size)` + step loop
- User confirmed visual parity after running the viewer twice — all presets, sliders, and keyboard shortcuts working correctly
- Phase 3 complete: CASimulator is now the single authoritative headless simulation path for both viewer and snap

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor viewer.py to delegate to CASimulator** - `ac8db84` (refactor)
2. **Task 2: Simplify __main__.py snap() to use CASimulator** - `54269ff` (refactor)
3. **Task 3: Visual verification** - user-verified, no code commit needed

## Files Created/Modified
- `plugins/cellular_automata/viewer.py` — Thin pygame display wrapper; 474 lines (was 1765); `self.simulator = CASimulator(preset, sim_size)`; run loop calls `self.simulator.step(dt)` for RGB array
- `plugins/cellular_automata/__main__.py` — Simplified snap(); 128 lines (was 256); snap() uses CASimulator directly in ~40 lines

## Decisions Made
- `ENGINE_LABELS` kept in viewer.py (display-only label mapping for HUD — not simulation logic, no reason to put in simulator.py)
- Paused state re-renders current frame via `self.simulator._render_frame(dt)` without advancing sim — keeps pause responsive without blanking the screen
- `_on_preset_select` calls `_rebuild_panel_if_needed()` after `apply_preset` to rebuild slider defs when engine type changes
- `FLOW_SLIDER_DEFS` kept in viewer.py (panel layout config, purely display-side)

## Deviations from Plan

None — plan executed exactly as written. The plan's specification of what to keep/remove was accurate and complete.

## Issues Encountered
- None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 is complete: CASimulator extracted and viewer delegating to it
- `CASimulator.render_float(dt)` returns `(H,W,3) float32 [0,1]` — exact interface Phase 4 (Scope plugin) will call
- Phase 4 (Scope plugin wrapper) can now import CASimulator and call `render_float` each frame without any pygame dependency

## Self-Check: PASSED

All artifacts verified:
- `plugins/cellular_automata/viewer.py` — FOUND (474 lines, no simulation methods)
- `plugins/cellular_automata/__main__.py` — FOUND (128 lines, snap() ~40 lines)
- `.planning/phases/03-extract-casimulator/03-02-SUMMARY.md` — FOUND (this file)
- Commit `ac8db84` (Task 1) — FOUND
- Commit `54269ff` (Task 2) — FOUND

---
*Phase: 03-extract-casimulator*
*Completed: 2026-02-18*
