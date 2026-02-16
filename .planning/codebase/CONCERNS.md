# Codebase Concerns

**Analysis Date:** 2026-02-16

## Tech Debt

**Missing Error Handling in Slider Components:**
- Issue: No validation for division by zero in `_val_to_x()` and `_x_to_val()` when min/max are equal
- Files: `plugins/cellular_automata/controls.py` (lines 57-67)
- Impact: If a slider is created with `min_val == max_val`, calculations fail with ZeroDivisionError
- Fix approach: Add guard clause `if self.max_val <= self.min_val: return self.min_val` and similar safeguards

**Unvalidated Mouse Coordinates in Viewer:**
- Issue: Mouse coordinates not bounds-checked before converting to simulation indices
- Files: `plugins/cellular_automata/viewer.py` (lines 519-531)
- Impact: Invalid negative indices or out-of-bounds access if mouse is off-canvas during painting
- Fix approach: Add explicit bounds validation: `sx = max(0, min(self.sim_size-1, sx))` before add_blob/remove_blob calls

**Magic Numbers in LFO System:**
- Issue: Hard-coded constants (0.012, 0.12, 0.8, etc.) scattered throughout state-coupled oscillator
- Files: `plugins/cellular_automata/viewer.py` (lines 420-437)
- Impact: Parameters are difficult to tune and understand; no way to export/reload configurations
- Fix approach: Create an LFO configuration dict in presets or a separate config system

**Float Format Inconsistencies:**
- Issue: Slider format strings vary inconsistently (.3f, .2f, .0f, .4f) with no central definition
- Files: `plugins/cellular_automata/viewer.py` (lines 279-308), `plugins/cellular_automata/controls.py` (line 44)
- Impact: UI display inconsistency, harder to standardize precision across engines
- Fix approach: Define FORMAT_PRESETS dict mapping parameter types to format strings

## Known Bugs

**Potential NaN in Color Normalization:**
- Symptoms: Strobing or color artifacts when edge/velocity maxima are very small
- Files: `plugins/cellular_automata/color_layers.py` (lines 132-135, 143-146)
- Trigger: Early in simulation when structure is sparse; edge_max_smooth can remain near zero
- Workaround: Current code has `norm = max(self._edge_max_smooth, 0.01)` which partially mitigates, but relies on initialization

**Mouse Painting Doesn't Respect Panel Bounds:**
- Symptoms: Painting continues even if mouse moves over control panel on the right
- Files: `plugins/cellular_automata/viewer.py` (lines 583-584)
- Trigger: User paints near canvas/panel boundary
- Workaround: Code checks `mx >= self.canvas_w` but only returns; it should be more precise about panel overlap

**Screenshot Path Traversal:**
- Symptoms: Path construction using `os.path.dirname()` three times assumes fixed directory structure
- Files: `plugins/cellular_automata/viewer.py` (lines 534-536)
- Trigger: If plugin directory structure changes or runs from different location, creates screenshots in wrong place
- Workaround: Paths work correctly in standard installation, but fragile

## Performance Bottlenecks

**FFT Kernel Recomputation Not Optimized:**
- Problem: Lenia `_build_kernel()` is called on every R or kernel shape parameter change, but FFT padding is not memoized
- Files: `plugins/cellular_automata/lenia.py` (lines 49-77)
- Cause: Each `set_params()` call with kernel changes triggers full kernel rebuild including zero-padding and FFT
- Improvement path: Pre-allocate padded kernel buffer, reuse across steps; only rebuild FFT when necessary

**Color Layer Computation Creates Temporary Arrays:**
- Problem: `compute_signals()` uses `np.roll()` four times and multiple temporary arrays for gradient computation
- Files: `plugins/cellular_automata/color_layers.py` (lines 124-130)
- Cause: Finite differences computed with roll instead of in-place operations
- Improvement path: Use pre-allocated buffers and in-place numpy operations; consider scipy.ndimage.sobel for gradients

**Memory Allocation in Loop:**
- Problem: `life.py` allocates new arrays in `step()` for every generation
- Files: `plugins/cellular_automata/life.py` (line 81: `new_cells = np.zeros_like(self.cells)`)
- Cause: No buffer reuse between steps
- Improvement path: Allocate write buffer in `__init__()`, swap buffers instead of creating new

**Unsmoothed FPS History:**
- Problem: FPS calculation stores every frame in list (max 30), then computes mean - spiky
- Files: `plugins/cellular_automata/viewer.py` (lines 628-632)
- Cause: No exponential moving average for stable FPS display
- Improvement path: Use single EMA float instead of list accumulation

## Fragile Areas

**Viewer LFO State Coupling:**
- Files: `plugins/cellular_automata/viewer.py` (lines 82-89, 213-225)
- Why fragile: LFO base values (`_lfo_base_mu`, `_lfo_base_sigma`, `_lfo_base_T`) are initialized from preset but can drift. If user adjusts slider manually, base must be updated (line 333-335) or LFO will pull wrong direction. If user switches presets, bases must reset from preset dict, not engine state (documented fix at line 213-215).
- Safe modification: When implementing new LFO features, always update bases from `get_preset()` return value, never from `self.engine` attributes. Add assertions: `assert self._lfo_base_mu == preset.get("mu")` on preset load.
- Test coverage: No unit tests for LFO; manual testing only

**Preset-Engine Mismatch Detection:**
- Files: `plugins/cellular_automata/viewer.py` (line 184-185), `plugins/cellular_automata/presets.py` (lines 216-223)
- Why fragile: `get_preset()` returns None silently if key not found. No validation that preset's `engine` field matches an actual engine. If typo in preset dict engine name, viewer creates wrong engine or crashes.
- Safe modification: Add validation in `get_preset()` to assert `preset["engine"] in ENGINE_CLASSES`. Add unit tests for all preset validity.
- Test coverage: No validation of preset structure

**Panel Button Index Out of Bounds:**
- Files: `plugins/cellular_automata/viewer.py` (lines 353-356)
- Why fragile: `_on_preset_select()` assumes `idx < len(preset_keys)` without checking. If ButtonRow logic is changed to send invalid indices, crash occurs.
- Safe modification: Add explicit bounds check: `if 0 <= idx < len(preset_keys): ...`
- Test coverage: Button selection not tested; only manual testing

**Seed Type Dispatch Without Validation:**
- Files: `plugins/cellular_automata/lenia.py` (lines 133-142), `plugins/cellular_automata/life.py` (line 176+)
- Why fragile: `seed()` method uses string dispatch with if/elif but no error for unknown types - silently falls through to default. If preset specifies invalid seed type, behavior is undefined.
- Safe modification: Raise ValueError for unknown seed_type: `else: raise ValueError(f"Unknown seed type: {seed_type}")`
- Test coverage: No unit tests for seed types

## Scaling Limits

**Simulation Resolution Scaling:**
- Current capacity: Tested up to 1024x1024 with Lenia (line 62: `sim_size` parameter)
- Limit: 2048x2048 starts hitting memory/performance wall on typical GPU (FFT overhead grows as O(N^2 log N))
- Scaling path: Implement adaptive resolution (lower res at high speed), use CUDA for FFT if available, consider hierarchical computation

**Panel Widget Layout:**
- Current capacity: Approximately 15-20 sliders fit in 300px wide panel before scrolling needed
- Limit: Adding more engines or parameters causes panel overflow with no scroll support
- Scaling path: Implement scrollable panel area, or collapse/expand engine sections

**Color Layer System:**
- Current capacity: 4 fixed layers with no easy way to add more
- Limit: Would require changes to LAYER_DEFS, weight array initialization, composite algorithm
- Scaling path: Generalize to N-layer system with dynamic allocation

## Test Coverage Gaps

**No Unit Tests for Engine Core Logic:**
- What's not tested: Lenia kernel generation, life rule parsing, Gray-Scott integration stability, Excitable threshold logic
- Files: All engine implementations `plugins/cellular_automata/{lenia,life,gray_scott,excitable}.py`
- Risk: Parameter errors or numerical bugs go undetected until visual inspection
- Priority: High - engine correctness is critical

**No Integration Tests for Viewer State Transitions:**
- What's not tested: Preset switching, engine switching, LFO state management, panel widget synchronization
- Files: `plugins/cellular_automata/viewer.py`
- Risk: State drift bugs (like the preset drift bug mentioned in MEMORY.md) could reoccur
- Priority: High - state management is complex

**No Validation Tests for Input Parameters:**
- What's not tested: Slider bounds validation, preset structure validity, seed type existence, invalid parameter combinations
- Files: `plugins/cellular_automata/` all modules
- Risk: Silent failures or crashes from user input
- Priority: Medium - catches edge cases

**No Performance/Benchmark Tests:**
- What's not tested: FPS targets, memory usage, FFT overhead, rendering latency
- Files: All performance-critical paths
- Risk: Performance regressions go undetected until user experience degrades
- Priority: Medium - helps track optimization efforts

## Security Considerations

**Unsanitized Screenshot Path:**
- Risk: If preset_key contains path traversal characters (e.g., `../../../etc/passwd`), could write outside screenshots dir
- Files: `plugins/cellular_automata/viewer.py` (line 540)
- Current mitigation: Preset keys are from hardcoded PRESETS dict, not user input
- Recommendations: Add explicit filename sanitization: `safe_name = preset_key.replace('/', '_').replace('..', '_')`; use Path library for path operations

**No Bounds Checking on Engine Configuration:**
- Risk: Negative R values, invalid neighborhood types, or extreme diffusion coefficients could cause numerical instability
- Files: All `set_params()` methods
- Current mitigation: Parameters come from UI sliders with defined ranges; presets are hardcoded
- Recommendations: Add parameter validation in `set_params()` with explicit min/max checks and error messages

## Missing Critical Features

**No Pause/Resume State Preservation:**
- Problem: Pausing clears simulation speed accumulator (line 211, 390) but doesn't preserve particle state effectively
- Blocks: Cannot reliably pause-and-examine without losing simulation flow

**No Undo/Redo for Manual Painting:**
- Problem: User can paint on canvas but cannot undo strokes
- Blocks: Experimentation limited; user must reset entire world

**No Export Video/Animation:**
- Problem: Screenshots save individual frames but no built-in video export
- Blocks: Cannot share results as animation without external tools

**No Parameter History/Animation:**
- Problem: Cannot create animations that sweep through parameter values over time
- Blocks: Advanced creative workflows limited

## Dependencies at Risk

**pygame-ce Requirement Not Pinned:**
- Risk: `pygame>=2.5.0` in requirements.txt allows breaking changes in future versions
- Impact: UI rendering could break with pygame 3.0+
- Migration plan: Pin to specific tested version or implement compatibility layer; consider switching to modern alternatives (pyglet, custom OpenGL renderer)

**numpy Version Flexibility:**
- Risk: `numpy>=1.24.0` is relatively loose; float64 behavior could change in major updates
- Impact: Numerical simulation results might diverge
- Migration plan: Test with numpy 2.0+; pin to known-good version

**No torch Dependency for Example Plugin:**
- Risk: Example plugin assumes torch but doesn't auto-install with plugin
- Impact: Plugin fails to import if torch not in environment
- Migration plan: Move torch import to lazy load or add as optional dependency in pyproject.toml

---

*Concerns audit: 2026-02-16*
