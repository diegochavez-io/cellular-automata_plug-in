# Testing Patterns

**Analysis Date:** 2026-02-16

## Test Framework

**Runner:**
- Not detected
- No test files found in codebase (`*test*.py`, `test_*.py`, `*_test.py`)
- No pytest.ini, conftest.py, tox.ini, or test configuration files

**Assertion Library:**
- Not applicable (no tests present)

**Run Commands:**
```bash
# No standard test execution pattern detected
# Code appears to be manually tested via the interactive viewer
# Command: python -m cellular_automata [preset]
# Or: python -m cellular_automata --list (lists all presets)
```

## Test File Organization

**Location:**
- No test directory exists
- No separate `tests/` folder
- No co-located test files (e.g., `viewer_test.py` next to `viewer.py`)

**Naming:**
- Not applicable (no tests)

**Structure:**
- Not applicable (no tests)

## Manual Testing Approach

**Current Pattern:**
The codebase relies on interactive manual testing through the pygame viewer. Tests are implicit through user interaction:

1. **Engine Tests** - Preset execution
   - Each preset in `presets.py` has been manually tested with the viewer
   - Presets known to work: "orbium", "geminium", "scutium", "aquarium", "mitosis", "dual_ring", "coral", "cardiac", "primordial", "classic_life", "highlife", "gs_maze"
   - User switches presets via number keys (1-9) or preset buttons in control panel

2. **Feature Tests** - Interactive controls
   - Parameter sliders tested via control panel (mu, sigma, R, T, layer weights, speed, brush size)
   - Mouse painting (left-click adds matter, right-click erases)
   - Keyboard controls: SPACE (pause), R (reset), S (screenshot), H (HUD toggle), TAB (panel toggle), F (fullscreen), Q (quit)
   - Engine switching tested via ENGINE buttons in control panel

3. **Rendering Tests** - Visual verification
   - Color layer compositing verified by eye (Core, Halo, Spark, Memory layers)
   - Hue rotation verified (~144s per full rotation)
   - Screenshot function `_save_screenshot()` tested by saving images

## Code Validation Patterns

Instead of formal tests, the codebase uses:

**Precondition Validation:**
```python
# From color_layers.py, line 121 (compute_signals)
np.clip(w, 0.0, 1.0, out=S[0])  # Ensure world values in [0, 1]

# From viewer.py, line 565
dt = min(now - last_time, 0.1)  # Cap dt to avoid jumps
```

**Bounds Checking:**
```python
# From controls.py, line 63-64 (Slider._x_to_val)
frac = max(0, min(1, frac))  # Clamp to [0, 1] before converting
val = self.min_val + frac * (self.max_val - self.min_val)

# From viewer.py, line 445-451 (LFO bounds with bounce)
if new_mu < mu_lo:
    new_mu = mu_lo
    self._mu_vel = abs(self._mu_vel) * 0.3  # Energy loss on bounce
```

**Defensive Initialization:**
```python
# From viewer.py, lines 98-99
preset = get_preset(start_preset)
self.engine_name = preset["engine"] if preset else "lenia"  # Fallback if preset not found

# From presets.py (dictionary with defaults)
PRESETS = {
    "orbium": {"engine": "lenia", "R": 13, "T": 10, ...}
}
```

**State Consistency:**
```python
# From viewer.py, lines 213-225 (apply_preset)
# LFO bases reset from preset definition, not engine state
# This prevents "preset drift bug" mentioned in MEMORY.md
self._lfo_base_mu = preset.get("mu", 0.15)
self._lfo_base_sigma = preset.get("sigma", 0.017)
self._lfo_base_T = preset.get("T", 10)
```

## No Formal Mocking

**Framework:** Not applicable (no tests present)

**Patterns:** Not applicable

**What to Mock:** Not applicable

**What NOT to Mock:** Not applicable

## Testing Considerations for New Code

When adding tests to this codebase, follow these patterns:

### Unit Test Pattern (if implemented)
```python
# Example structure for Lenia engine tests
def test_lenia_step_updates_generation():
    lenia = Lenia(size=256, R=13, T=10)
    lenia.seed("random")
    initial_gen = lenia.generation
    lenia.step()
    assert lenia.generation == initial_gen + 1

def test_lenia_growth_function():
    lenia = Lenia(mu=0.15, sigma=0.017)
    # Bell curve should peak at mu
    peak = lenia.growth(0.15)  # At mu
    sides = lenia.growth(0.05)  # Far from mu
    assert peak > sides

def test_bounds_enforcement():
    lenia = Lenia(size=256)
    lenia.world[:] = 2.0  # Out of bounds
    lenia.step()
    assert lenia.world.max() <= 1.0
    assert lenia.world.min() >= 0.0
```

### Integration Test Pattern (if implemented)
```python
# Example: Test viewer with preset
def test_viewer_loads_preset():
    viewer = Viewer(start_preset="orbium")
    assert viewer.engine_name == "lenia"
    assert viewer.preset_key == "orbium"
    assert viewer.engine.world.shape == (1024, 1024)

def test_viewer_switches_presets():
    viewer = Viewer(start_preset="orbium")
    initial_engine = viewer.engine_name

    viewer._apply_preset("classic_life")
    assert viewer.engine_name == "life"
    assert viewer.preset_key == "classic_life"

def test_lfo_oscillation():
    viewer = Viewer(start_preset="coral")
    # LFO should modulate mu over time
    initial_mu = viewer.engine.mu
    for _ in range(60):  # ~1 second at 60fps
        viewer._apply_lfo(dt=0.016)
    # mu should have changed (but not by much in 1 second)
    assert viewer.engine.mu != initial_mu
```

### Rendering Test Pattern (if implemented)
```python
# Example: Test color layer compositing
def test_color_compositing():
    layers = ColorLayerSystem(size=512)
    world = np.random.rand(512, 512).astype(np.float32)

    signals = layers.compute_signals(world)
    rgb = layers.composite(signals)

    assert rgb.shape == (512, 512, 3)
    assert rgb.dtype == np.uint8
    assert rgb.min() >= 0
    assert rgb.max() <= 255

def test_layer_weights_affect_output():
    layers = ColorLayerSystem(size=256)
    world = np.ones((256, 256), dtype=np.float32) * 0.5

    signals = layers.compute_signals(world)

    # Zero out all weights except first layer
    layers.weights[:] = [1.0, 0.0, 0.0, 0.0]
    rgb1 = layers.composite(signals)

    # Zero out all weights except second layer
    layers.weights[:] = [0.0, 1.0, 0.0, 0.0]
    rgb2 = layers.composite(signals)

    # Outputs should differ
    assert not np.array_equal(rgb1, rgb2)
```

## Special Notes on Testability

**Strengths:**
- Clean separation of concerns: Engine classes inherit from `CAEngine` interface
- Stateless helper functions: `_hsv_to_rgb()`, `parse_rule()` easily unit-testable
- Configurable initialization: Engines accept size, parameters as constructor arguments
- Deterministic seeding: `seed("blobs")`, `seed("ring")` produce reproducible patterns

**Challenges:**
- Heavy numpy/scipy dependencies: Tests would need numpy fixtures
- Viewer tightly coupled to pygame: UI tests require mocking pygame or headless testing
- Large simulation state: 1024x1024 grids are expensive to compare in tests
- Continuous values: Float comparisons need tolerance (`np.allclose()`)

## Coverage Gaps

**Areas with no formal tests:**
- Engine parameter validation (mu, sigma bounds)
- LFO state-coupled oscillator math (`_apply_lfo()`)
- FFT-based convolution in Lenia (numerical correctness)
- Color layer feedback into simulation (`apply_feedback()`)
- Preset application and engine switching
- Mouse interaction and painting
- Screenshot saving
- UI control responsiveness

---

*Testing analysis: 2026-02-16*
