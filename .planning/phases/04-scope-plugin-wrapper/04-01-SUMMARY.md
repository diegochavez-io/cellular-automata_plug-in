---
phase: 04-scope-plugin-wrapper
plan: 01
subsystem: plugin
tags: [scope, cellular-automata, pytorch, numpy, scipy, hatchling, plugin-registration]

# Dependency graph
requires:
  - phase: 03-extract-casimulator
    provides: CASimulator headless class with render_float(), step(), set_runtime_params(), apply_preset()

provides:
  - pyproject.toml with hatchling build config, scope entry point, pygame file exclusions
  - plugin.py with dual-pattern @hookimpl + registry.register() registration
  - pipeline.py with dual-pattern CAPipeline: BasePipelineConfig+Pipeline ABC when Scope API available, plain class fallback otherwise
  - CASimulator.warmup parameter and run_warmup() method for deferred warmup
  - Complete Scope plugin package structure installable via uv run daydream-scope install

affects: [05-cupy-gpu-acceleration, 06-runpod-deployment]

# Tech tracking
tech-stack:
  added: [torch (tensor output), hatchling (build backend)]
  patterns:
    - Dual-pattern Scope API: try formal BasePipelineConfig+Pipeline ABC, fallback to plain class
    - Shared module-level helpers (_ca_init, _ca_call) assigned as class method bodies
    - Deferred warmup via warmup=False constructor flag + run_warmup() on first __call__()
    - Wall-clock dt via time.perf_counter() clamped to [0.001, 0.1]

key-files:
  created:
    - plugins/cellular_automata/pyproject.toml
    - plugins/cellular_automata/plugin.py
    - plugins/cellular_automata/pipeline.py
  modified:
    - plugins/cellular_automata/simulator.py

key-decisions:
  - "Dual-pattern for Scope API: try formal BasePipelineConfig+Pipeline ABC import, fall back to plain class — single codebase runs in Scope and bare Python"
  - "Deferred warmup: CASimulator(warmup=False) in __init__, run_warmup() on first __call__() — plugin loads instantly, warmup runs before first frame"
  - "Shared _ca_init/_ca_call module-level helpers assigned to both class branches — zero code duplication"
  - "Return dict {'video': tensor} not raw tensor — matches Scope pipeline contract"
  - "_PRESET_CHOICES module constant lists all 23 headless-safe presets by internal key"

patterns-established:
  - "Plugin dual-pattern: try Scope API imports, define class in if/else block based on _HAS_SCOPE_API flag"
  - "Deferred warmup: pass warmup=False to constructor, run_warmup() on first frame"
  - "All runtime params from kwargs every frame — never store from __init__"

requirements-completed: [PLUG-01, PLUG-02, PLUG-03, PLUG-04, PLUG-05, PLUG-06, PLUG-07, PLUG-08, PLUG-10]

# Metrics
duration: 6min
completed: 2026-02-18
---

# Phase 4 Plan 1: Scope Plugin Wrapper Summary

**Dual-pattern Scope plugin package (pyproject.toml + plugin.py + pipeline.py) with deferred warmup, THWC tensor output, wall-clock dt, and runtime kwargs — installable into Scope or runnable standalone**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-18T06:56:51Z
- **Completed:** 2026-02-18T07:02:54Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created complete Scope plugin package: pyproject.toml with hatchling build, cellular_automata entry point, and pygame file exclusions
- Implemented dual-pattern CAPipeline: BasePipelineConfig + Pipeline ABC when Scope API available, plain class with static ui_field_config() fallback when not
- Added warmup=True parameter to CASimulator.__init__() and run_warmup() public method for deferred warmup control
- Plugin loads in 0.066s (warmup deferred); all PLUG-01 through PLUG-10 requirements satisfied

## Task Commits

Each task was committed atomically:

1. **Task 1: Add warmup param to CASimulator, create pyproject.toml + plugin.py** - `c6df474` (feat)
2. **Task 2: Create pipeline.py with CAPipeline class** - `b36a7d5` (feat)

**Plan metadata:** _(to be added by final commit)_

## Files Created/Modified
- `plugins/cellular_automata/pyproject.toml` - Hatchling build config, scope entry point, pygame file exclusions (viewer.py, controls.py, __main__.py)
- `plugins/cellular_automata/plugin.py` - Dual-pattern plugin registration: @hookimpl when scope.core.plugins.hookspecs available, registry.register() fallback
- `plugins/cellular_automata/pipeline.py` - CAPipeline with dual-pattern (BasePipelineConfig+Pipeline ABC or plain class), shared _ca_init/_ca_call helpers, all runtime params from kwargs, THWC tensor output, wall-clock dt
- `plugins/cellular_automata/simulator.py` - Added warmup=True parameter, self._warmup guard in _apply_preset(), run_warmup() public method

## Decisions Made
- Dual-pattern for Scope API compatibility: try BasePipelineConfig+Pipeline ABC import at module load, define CAPipeline in if/else block — single file works in Scope and local Python
- Deferred warmup: plugin loads instantly (CASimulator(warmup=False) in __init__), run_warmup() fires on first __call__() before first frame
- Shared _ca_init/_ca_call module-level functions assigned as class __init__/__call__ in both branches — zero code duplication
- Return {"video": tensor} dict matching Scope pipeline contract (not raw tensor)
- Wall-clock dt clamped to [0.001, 0.1] to prevent runaway LFO on lag spikes

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed torch for local verification**
- **Found during:** Task 2 verification
- **Issue:** torch not installed in local Python 3.12 environment, verification script failed on `import torch`
- **Fix:** `python3 -m pip install torch --quiet` to enable local verification
- **Files modified:** system site-packages (not committed)
- **Verification:** All PLUG checks passed after install
- **Committed in:** N/A (environment change, not project file)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Torch install required for verification only — it's a declared dependency in pyproject.toml so it will be present in the Scope environment. No scope creep.

## Issues Encountered
- Scope formal API (scope.core.plugins.hookspecs, BasePipelineConfig, Pipeline) not installed locally — expected. Fallback branch verified to work correctly. Formal branch will activate when installed into Scope.

## User Setup Required
None - no external service configuration required. Plugin is ready to install via `uv run daydream-scope install plugins/cellular_automata/`.

## Next Phase Readiness
- Plugin package structure complete and verified working standalone
- CAPipeline produces correct THWC float32 tensors in [0,1]
- Ready for Phase 5: CuPy GPU acceleration (port numpy/scipy to cupyx)
- Ready for smoke test: `uv run daydream-scope install plugins/cellular_automata/` then load pipeline in Scope UI

---
*Phase: 04-scope-plugin-wrapper*
*Completed: 2026-02-18*

## Self-Check: PASSED

- FOUND: plugins/cellular_automata/pyproject.toml
- FOUND: plugins/cellular_automata/plugin.py
- FOUND: plugins/cellular_automata/pipeline.py
- FOUND: plugins/cellular_automata/simulator.py
- FOUND: .planning/phases/04-scope-plugin-wrapper/04-01-SUMMARY.md
- FOUND commit: c6df474 (Task 1)
- FOUND commit: b36a7d5 (Task 2)
- All PLUG verification assertions passed
