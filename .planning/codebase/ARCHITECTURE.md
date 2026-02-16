# Architecture

**Analysis Date:** 2026-02-16

## Pattern Overview

**Overall:** Plugin-Based Modular CA System with Layered Rendering

The codebase implements a dual-architecture design:
1. **Scope Integration Layer**: DayDream Scope plugin system for real-time video pipeline integration
2. **Standalone CA System**: Independent cellular automata engines with interactive visualization via Pygame

The architecture separates CA computation logic, rendering/visualization, and Scope integration, allowing independent evolution of each layer.

**Key Characteristics:**
- Polymorphic engine architecture with pluggable CA algorithms (Lenia, Life, Excitable, Gray-Scott)
- Multi-layer additive color rendering system with state-coupled oscillator feedback
- Event-driven interactive controls via Pygame with fractional time stepping
- Generic preset-based configuration enabling zero-code engine switching
- Abstracted parameter management (load-time vs runtime) for Scope UI integration

## Layers

**Computation Layer (Engines):**
- Purpose: Execute cellular automata simulation logic (FFT convolution, growth functions, state updates)
- Location: `plugins/cellular_automata/lenia.py`, `life.py`, `gray_scott.py`, `excitable.py`
- Contains: Engine-specific algorithms, kernel building, state mutation, parameter management
- Depends on: `engine_base.py` (abstract interface), NumPy for compute
- Used by: Viewer main loop, Color layer system for feedback
- Key abstraction: `CAEngine` base class with `step()`, `set_params()`, `seed()`, `get_params()`, `apply_feedback()`

**Color Rendering Layer:**
- Purpose: Transform simulation state into 4-layer additive RGB visualization with feedback computation
- Location: `plugins/cellular_automata/color_layers.py`
- Contains: HSV-to-RGB conversion, layer compositing, gradient computation, memory buffer, feedback generation, hue rotation
- Depends on: Computation layer (reads world state), NumPy
- Used by: Viewer for frame rendering and engine feedback
- Pattern: `ColorLayerSystem` accumulates 4 channels (Core, Halo, Spark, Memory) each with rotating rainbow hues

**Interactive Viewer Layer:**
- Purpose: SDL/Pygame event loop, UI control panel, real-time parameter adjustment, preset switching, save/load
- Location: `plugins/cellular_automata/viewer.py`
- Contains: Main event loop, control panel management, mouse/keyboard input handling, preset application, LFO oscillator
- Depends on: All engines, color layer system, controls module, presets
- Used by: End user interaction; can be standalone or Scope-integrated
- Pattern: State machine with mode tracking (pause, fullscreen, panel visibility) and fractional speed accumulation

**Control/UI Layer:**
- Purpose: Render minimal dark-themed UI widgets (sliders, buttons) directly in Pygame
- Location: `plugins/cellular_automata/controls.py`
- Contains: Slider, Button, ControlPanel classes with hit testing and value formatting
- Depends on: Pygame, THEME constants
- Used by: Viewer to build and manage parameter controls

**Configuration Layer:**
- Purpose: Define preset parameters and metadata for all CA engine configurations
- Location: `plugins/cellular_automata/presets.py`
- Contains: PRESETS dict with engine-specific parameter sets, seed types, descriptions; PRESET_ORDER list for UI ordering; helper functions
- Depends on: None (data-only)
- Used by: Viewer, example_plugin for preset loading

**Scope Plugin Integration:**
- Purpose: Expose cellular automata as a DayDream Scope pipeline for real-time video input processing
- Location: `plugins/cellular_automata/plugin.py`, `plugins/example_plugin/plugin.py` (template)
- Contains: `register_pipelines()` hook, pipeline class registration
- Depends on: Engine architecture, Scope SDK (imported at runtime by Scope)
- Used by: Scope runtime when plugin is installed

## Data Flow

**Interactive Viewer Loop (Standalone):**

```
1. [Initialization] Load preset → Create engine → Seed world → Initialize color layers
2. [Main Loop]
   a. [Input] Poll Pygame events → Mouse/keyboard → Update UI controls
   b. [Simulation] Accumulate fractional time steps:
      - speed_accumulator += sim_speed
      - While accumulator >= 1.0: engine.step(), accumulator -= 1.0
   c. [Modulation] Update LFO phase → Modulate Lenia mu/sigma/T based on organism mass (state-coupled oscillator)
   d. [Feedback] Compute color layers → Generate feedback field → engine.apply_feedback(feedback)
   e. [Rendering] Color layers compose RGBA frame → Convert to pygame.Surface → Draw to screen
   f. [UI] Draw control panel with current slider values
   g. [Output] Present frame (vsync)
   h. [Capture] Optional: Save screenshot to timestamped PNG
3. [Termination] On Q/ESC: Close window, exit loop
```

**State Progression Per Frame:**
```
Engine world (float64 [0,1])
  ↓ [step()] via FFT convolution
Neighborhood potential (U)
  ↓ [growth()] Gaussian mapping
Growth rate [-1, 1]
  ↓ [clip to [0,1]] Integration with dt
Updated world state
  ↓ [apply_feedback()] Add color layer feedback
World with color influence

World → ColorLayerSystem.process()
  ├─ Core layer: world density
  ├─ Halo layer: gradient magnitude
  ├─ Spark layer: temporal velocity
  ├─ Memory layer: EMA history
  ↓ [additive composite with HSV rotation]
RGBA frame [0, 255]
```

**State Management:**

- **Engine State**: Mutable world array (continuously updated), generation counter, kernel cache (rebuilt on param change)
- **Viewer State**: Active preset, engine instance, speed accumulator, LFO phase/velocity/mass tracking, pause flag, panel visibility
- **Color State**: Layer weights, master feedback scale, hue rotation time, memory buffer with decay
- **UI State**: Slider values (cached references to engine/viewer params), button selections

Critical: **LFO bases (mu, sigma, T) are read from preset dict, NOT from engine state**, to prevent drift across preset reloads.

## Key Abstractions

**CAEngine Base Class:**
- Purpose: Define the contract all CA engines must implement
- Location: `plugins/cellular_automata/engine_base.py`
- Pattern: Abstract base class with concrete helper methods (add_blob, remove_blob, stats property)
- Concrete implementations: `Lenia`, `Life`, `Excitable`, `GrayScott`

**Viewer Preset Application:**
- Purpose: Encapsulate the logic for switching presets, creating engines, seeding, and syncing UI
- Location: `plugins/cellular_automata/viewer.py:_apply_preset()`
- Pattern: Distinguish engine-changed vs param-only updates; preserve LFO bases from preset dict

**Fractional Speed System:**
- Purpose: Decouple display FPS from simulation speed; enable smooth sub-1.0x speeds
- Location: `plugins/cellular_automata/viewer.py` (speed_accumulator pattern)
- Pattern: Accumulate real speed value; step() only when accumulator >= 1.0; prevents frame strobing

**State-Coupled Oscillator (Lenia only):**
- Purpose: Create organic morphing by modulating growth parameters based on organism mass feedback
- Location: `plugins/cellular_automata/viewer.py` (LFO state machine)
- Pattern: mu drifts right (predator behavior), sigma drifts left (prey behavior), mass provides delayed feedback creating cycle

**Color Layer Feedback:**
- Purpose: Enable visual elements to influence simulation and create visual loops
- Location: `plugins/cellular_automata/color_layers.py:compute_feedback()`
- Pattern: Each layer (Core, Halo, Spark, Memory) has independent feedback coefficient; additive blending

## Entry Points

**Standalone Viewer:**
- Location: `plugins/cellular_automata/__main__.py`
- Triggers: `python -m cellular_automata [preset] [--size N] [--window WxH]`
- Responsibilities: Parse command-line args, instantiate Viewer, run event loop, clean up pygame

**Scope Plugin:**
- Location: `plugins/cellular_automata/plugin.py` and `plugins/example_plugin/plugin.py`
- Triggers: DayDream Scope runtime calls `register_pipelines(registry)` on plugin load
- Responsibilities: Instantiate pipeline class, register with Scope, manage parameter mapping

**Pipeline Class (for Scope):**
- Template: `plugins/example_plugin/pipeline.py` shows three patterns:
  - `ExamplePipeline`: Dual text + video input (implements `prepare()` for video-input mode)
  - `TextOnlyPipeline`: Text prompts only (no `prepare()`)
  - `VideoOnlyPipeline`: Video frames only (requires `prepare()`)
- Critical: Runtime parameters read from `kwargs` in `__call__()`, NOT stored in `__init__()`

## Error Handling

**Strategy:** Defensive parameter validation with graceful degradation

**Patterns:**
- Invalid preset name: Return None, use fallback engine
- Missing preset field: Use hardcoded defaults (e.g., R=13 for Lenia)
- Invalid slider input: Clamp to [min, max] range
- Engine creation failure: Print error, continue with existing engine
- Corrupted file save/load: Skip without crashing

## Cross-Cutting Concerns

**Logging:** Console output via `print()` for startup messages, engine creation, preset changes. No structured logging framework used.

**Validation:**
- Parameter bounds checked in slider handlers (clamp before apply)
- Engine state validated on set_params() (rebuild kernel only if needed to avoid waste)
- World clipping to [0, 1] after every update step

**State Synchronization:**
- Sliders stay in sync via `_sync_sliders_from_engine()` which reads current engine.get_params() and updates slider display
- Preset LFO bases cached at load time to avoid drift
- Color layer state reset on every preset change to prevent color bleed

**Performance:**
- FFT-based convolution in Lenia (O(N log N) per step vs O(N*R^2) direct)
- Pre-allocated numpy buffers in color layer (avoid GC pressure)
- Float32 for color layers, float64 for simulation (balance precision vs speed)
- Smoothed max normalization (EMA) prevents strobing in visualization

---

*Architecture analysis: 2026-02-16*
