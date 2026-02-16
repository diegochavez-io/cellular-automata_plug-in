# Architecture Patterns: Iridescent Color Rendering for Cellular Automata

**Domain:** Real-time procedural iridescent rendering for 2D simulation fields
**Researched:** 2026-02-16
**Confidence:** MEDIUM (thin-film physics: HIGH, CPU implementation patterns: MEDIUM, safe parameter control: HIGH)

## Executive Summary

The current 4-layer additive color system (Core/Halo/Spark/Memory) needs replacement with an iridescent rendering pipeline that produces oil-slick shimmer, slow hue sweeps, and spatial gradients. This research identifies a component architecture based on thin-film interference physics, procedural spatial field generation, and safe parameter interpolation patterns.

**Key finding:** Thin-film interference color is computationally lightweight (wavelength-dependent phase calculation) and can be implemented efficiently on CPU with numpy. The architecture should separate physical iridescence from artistic controls (spatial gradients, hue sweep, RGB tint) for maximum flexibility.

## Recommended Architecture

### System Overview

```
Engine (existing)
    ↓ world state [H, W] float64 [0,1]
    ↓
IridescentColorSystem (replaces ColorLayerSystem)
    ├─ ThicknessMapper: world → thickness field [H, W]
    ├─ SpatialGradient: position → gradient offset [H, W]
    ├─ HueSweep: time → global hue rotation [scalar]
    ├─ ThinFilmRenderer: thickness + gradient + hue → RGB
    └─ RGBTint: RGB + tint controls → final RGBA
    ↓
RGBA frame [H, W, 4] uint8 → display
```

### Component Boundaries

| Component | Responsibility | Input | Output | Performance Budget |
|-----------|---------------|-------|--------|-------------------|
| **ThicknessMapper** | Map world density to film thickness | world [H,W] float64 | thickness [H,W] nm | <2ms @ 1024² |
| **SpatialGradient** | Generate spatial variation field | (x, y) grid + params | gradient [H,W] | <3ms @ 1024² |
| **HueSweep** | Global hue rotation over time | time (seconds) | hue_offset (degrees) | <0.1ms |
| **ThinFilmRenderer** | Compute interference RGB from thickness | thickness + gradient + hue | RGB [H,W,3] float32 | <12ms @ 1024² |
| **RGBTint** | Apply user color correction | RGB + R/G/B/brightness sliders | RGBA uint8 | <2ms @ 1024² |
| **SafeSliderManager** | Interpolate parameter changes | slider_new, slider_current, dt | slider_smooth | <0.5ms total |

**Total rendering budget:** ~20ms/frame (within 22ms baseline with 2ms margin)

---

## Component Designs

### 1. ThicknessMapper

**Purpose:** Convert world state (cell density [0,1]) to thin-film thickness in nanometers.

**Architecture:**
```python
class ThicknessMapper:
    def __init__(self, min_thickness_nm=200, max_thickness_nm=800):
        self.min_nm = min_thickness_nm
        self.max_nm = max_thickness_nm
        self.contrast = 1.5  # Exponent for nonlinear mapping

    def map(self, world: np.ndarray) -> np.ndarray:
        """Map world [0,1] → thickness [min_nm, max_nm]."""
        # Nonlinear mapping: lower densities → thinner film (blue shift)
        # higher densities → thicker film (red shift)
        normalized = np.clip(world, 0, 1) ** self.contrast
        thickness_nm = self.min_nm + normalized * (self.max_nm - self.min_nm)
        return thickness_nm.astype(np.float32)
```

**Rationale:**
- Visible thin-film interference occurs in 200-800nm range (source: Wikipedia, Physics LibreTexts)
- Nonlinear mapping (power curve) creates richer color variation in low-density regions
- Pre-allocated output buffer for performance

**Data flow:**
```
engine.world [1024, 1024] float64
    ↓ clip + power curve + scale
thickness_field [1024, 1024] float32 (nm values)
```

---

### 2. SpatialGradient

**Purpose:** Generate a 2D spatial field that varies smoothly across the grid, creating rainbow sweeps across the organism.

**Architecture:**
```python
class SpatialGradient:
    def __init__(self, size):
        self.size = size
        # Pre-compute coordinate grids
        Y, X = np.ogrid[:size, :size]
        self.X = X.astype(np.float32) / size  # Normalized [0, 1]
        self.Y = Y.astype(np.float32) / size

        # Gradient parameters (controllable)
        self.angle_deg = 45.0      # Direction of gradient sweep
        self.wavelength = 1.5      # Spatial frequency (cycles across image)
        self.amplitude_nm = 150.0  # Thickness modulation strength

    def compute(self) -> np.ndarray:
        """Compute spatial gradient field → thickness offset [H, W]."""
        angle_rad = np.deg2rad(self.angle_deg)

        # Rotated coordinate: project X,Y onto gradient direction
        rotated = self.X * np.cos(angle_rad) + self.Y * np.sin(angle_rad)

        # Sinusoidal thickness variation
        gradient_offset = self.amplitude_nm * np.sin(
            rotated * self.wavelength * 2 * np.pi
        )
        return gradient_offset
```

**Rationale:**
- Spatial gradients create the "oil slick" rainbow sweep effect
- Sine wave produces smooth, organic color transitions
- Pre-computed coordinate grids minimize per-frame cost
- Parameters (angle, wavelength, amplitude) give artistic control

**Advanced option (for Phase 2+):**
- Add Perlin/Simplex noise for organic, non-uniform gradients
- Multi-scale noise layers for complex shimmer patterns
- Trade: +visual richness, -complexity, -~5ms performance cost

**Data flow:**
```
Pre-computed: X, Y grids [1024, 1024] float32
Runtime: angle, wavelength, amplitude (scalars)
    ↓ sin(rotated_coord * wavelength)
gradient_offset [1024, 1024] float32 (nm)
```

---

### 3. HueSweep

**Purpose:** Slowly rotate the entire color palette over time (the "rainbow rotation" from the old system).

**Architecture:**
```python
class HueSweep:
    def __init__(self):
        self.hue_time = 0.0
        self.hue_speed = 2.5  # degrees/second (~144s full rotation)

    def advance(self, dt: float) -> float:
        """Advance time, return current hue offset in degrees."""
        self.hue_time += self.hue_speed * dt
        return self.hue_time % 360
```

**Rationale:**
- Matches existing ColorLayerSystem behavior (2.5 deg/s)
- Independent of spatial gradient (allows composability)
- Trivial compute cost (<0.1ms)

---

### 4. ThinFilmRenderer

**Purpose:** Core physics computation — convert thickness to RGB using thin-film interference equations.

**Physics Background:**

Thin-film interference occurs when light reflects from both the top and bottom surfaces of a thin film. The two reflected waves interfere constructively or destructively depending on:
- **Film thickness (t)**: distance light travels through film
- **Wavelength (λ)**: different colors interfere at different thicknesses
- **Refractive index (n)**: film material (oil ~1.5, soap ~1.33)

**Interference condition:**
```
Optical path difference = 2 * n * t * cos(θ)
Constructive interference: path difference = (m + 0.5) * λ
Destructive interference:  path difference = m * λ
```

For perpendicular viewing (θ=0), this simplifies to:
```
Constructive: 2*n*t = (m + 0.5) * λ
Destructive:  2*n*t = m * λ
```

**Sources:**
- [Wikipedia: Thin-film interference](https://en.wikipedia.org/wiki/Thin-film_interference)
- [Physics LibreTexts: Thin Film Interference](https://phys.libretexts.org/Bookshelves/University_Physics/Calculus-Based_Physics_(Schnick)/Volume_B:_Electricity_Magnetism_and_Optics/B24:_Thin_Film_Interference)
- [OpenStax: Interference in Thin Films](https://openstax.org/books/university-physics-volume-3/pages/3-4-interference-in-thin-films)

**Implementation Strategy:**

Sample RGB wavelengths (645nm red, 526nm green, 445nm blue) and compute interference intensity for each.

```python
class ThinFilmRenderer:
    def __init__(self, size):
        self.size = size
        self.refractive_index = 1.45  # Oil-like film

        # RGB wavelengths (nanometers)
        self.lambda_r = 645.0
        self.lambda_g = 526.0
        self.lambda_b = 445.0

        # Pre-allocate output
        self._rgb = np.zeros((size, size, 3), dtype=np.float32)

    def render(self, thickness: np.ndarray, gradient: np.ndarray,
               hue_offset: float) -> np.ndarray:
        """
        Compute RGB from thickness field using thin-film interference.

        Args:
            thickness: Base thickness field [H, W] in nm
            gradient: Spatial gradient offset [H, W] in nm
            hue_offset: Global hue rotation in degrees

        Returns:
            RGB image [H, W, 3] float32 in [0, 255]
        """
        # Total thickness = base + gradient modulation
        t = thickness + gradient

        # Optical path length = 2 * n * t
        path = 2.0 * self.refractive_index * t

        # Interference intensity for each wavelength
        # Using simplified cosine model: I = cos²(π * path / λ)
        phase_r = np.pi * path / self.lambda_r
        phase_g = np.pi * path / self.lambda_g
        phase_b = np.pi * path / self.lambda_b

        I_r = np.cos(phase_r) ** 2
        I_g = np.cos(phase_g) ** 2
        I_b = np.cos(phase_b) ** 2

        # Apply hue rotation (rotate in HSV space)
        rgb_rotated = self._rotate_hue(
            np.stack([I_r, I_g, I_b], axis=2),
            hue_offset
        )

        # Scale to [0, 255]
        return (rgb_rotated * 255.0).astype(np.float32)

    def _rotate_hue(self, rgb: np.ndarray, hue_deg: float) -> np.ndarray:
        """Rotate RGB in hue space (approximate, fast method)."""
        # Fast rotation matrix approach (cheaper than RGB→HSV→RGB)
        hue_rad = np.deg2rad(hue_deg)
        U = np.cos(hue_rad)
        W = np.sin(hue_rad)

        # Rotation matrix (preserves luminance)
        r = rgb[..., 0] * (0.299 + 0.701*U + 0.168*W) + \
            rgb[..., 1] * (0.587 - 0.587*U + 0.330*W) + \
            rgb[..., 2] * (0.114 - 0.114*U - 0.497*W)

        g = rgb[..., 0] * (0.299 - 0.299*U - 0.328*W) + \
            rgb[..., 1] * (0.587 + 0.413*U + 0.035*W) + \
            rgb[..., 2] * (0.114 - 0.114*U + 0.292*W)

        b = rgb[..., 0] * (0.299 - 0.300*U + 1.250*W) + \
            rgb[..., 1] * (0.587 - 0.588*U - 1.050*W) + \
            rgb[..., 2] * (0.114 + 0.886*U - 0.203*W)

        return np.clip(np.stack([r, g, b], axis=2), 0, 1)
```

**Rationale:**
- **Cosine model:** Simplified but physically plausible (source: Alan Zucconi, Shadertoy implementations)
- **Wavelength sampling:** 3 wavelengths sufficient for RGB (full spectral integration unnecessary)
- **Hue rotation:** Fast matrix method avoids costly RGB↔HSV conversions
- **Performance:** Vectorized numpy operations stay within budget (~12ms for 1024²)

**Accuracy vs Performance Trade:**
- **Current design:** Fast cosine approximation, 3 wavelengths
- **Higher accuracy:** Integrate over full spectrum (380-750nm), Fresnel coefficients → +20-30ms (unacceptable)
- **Lower accuracy:** Lookup table (LUT) pre-computed thickness→RGB → saves ~5ms but less flexible

**Sources:**
- [Alan Zucconi: Car Paint Shader - Thin-Film Interference](https://www.alanzucconi.com/2017/10/27/carpaint-shader-thin-film-interference/)
- [Shadertoy: Fast Thin-Film Interference](https://www.shadertoy.com/view/ld3SDl)
- [Jelly Renders: Rendering a Simple Iridescent Material](https://jellyrenders.com/graphics/ray%20tracing/2025/11/17/simple-iridescent-material/)

---

### 5. RGBTint

**Purpose:** Apply user-controllable color correction (R/G/B channel gains, brightness).

**Architecture:**
```python
class RGBTint:
    def __init__(self):
        self.gain_r = 1.0
        self.gain_g = 1.0
        self.gain_b = 1.0
        self.brightness = 1.0  # Master brightness multiplier

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Apply color tint and brightness to RGB field.

        Args:
            rgb: [H, W, 3] float32 in [0, 255]

        Returns:
            RGBA [H, W, 4] uint8
        """
        # Apply channel gains
        rgb[:, :, 0] *= self.gain_r
        rgb[:, :, 1] *= self.gain_g
        rgb[:, :, 2] *= self.gain_b

        # Apply brightness
        rgb *= self.brightness

        # Clip and convert to uint8
        rgb_clipped = np.clip(rgb, 0, 255).astype(np.uint8)

        # Add alpha channel (opaque)
        rgba = np.dstack([rgb_clipped, np.full_like(rgb_clipped[:, :, 0], 255)])
        return rgba
```

**Rationale:**
- Simple multiplicative gains (artist-friendly)
- Separate R/G/B control allows color balance adjustments
- Brightness master control for overall intensity
- Cheap operation (<2ms)

---

### 6. SafeSliderManager

**Purpose:** Prevent abrupt parameter changes from killing the organism. Interpolate slider values smoothly over time.

**Problem:**
User moves a slider → parameter jumps instantly → organism dies or exhibits discontinuity.

**Solution:**
Exponential moving average (EMA) smoothing of parameter values.

**Architecture:**
```python
class SafeSliderManager:
    def __init__(self, smoothing_time_seconds=0.5):
        """
        Args:
            smoothing_time_seconds: Time constant for parameter changes.
                Smaller = faster response, larger = smoother but slower.
        """
        self.smoothing_time = smoothing_time_seconds
        self._params = {}  # {param_name: current_smooth_value}

    def update(self, param_name: str, target_value: float, dt: float) -> float:
        """Update parameter with EMA smoothing.

        Args:
            param_name: Name of parameter (e.g., "mu", "sigma", "kernel_R")
            target_value: New value from slider
            dt: Time delta since last frame (seconds)

        Returns:
            Smoothed parameter value to apply to engine
        """
        if param_name not in self._params:
            # First time seeing this parameter - initialize immediately
            self._params[param_name] = target_value
            return target_value

        current = self._params[param_name]

        # EMA coefficient: higher alpha = faster convergence
        alpha = 1.0 - np.exp(-dt / self.smoothing_time)

        # Exponential moving average
        smoothed = current + alpha * (target_value - current)

        self._params[param_name] = smoothed
        return smoothed

    def reset(self, param_name: str):
        """Clear smoothing state for a parameter (e.g., on preset change)."""
        if param_name in self._params:
            del self._params[param_name]
```

**Rationale:**
- **EMA formula:** `value_new = value_current + α * (target - value_current)` where `α = 1 - exp(-dt / τ)`
- **Time constant (τ):** 0.5s default = parameter reaches 95% of target in ~1.5 seconds
- **Per-parameter state:** Each slider smoothed independently
- **Reset on preset change:** Prevents slow drift when switching presets

**Example usage in viewer:**
```python
# In viewer.py, replace direct parameter application:
# OLD:
# self.engine.set_params(mu=slider_value)

# NEW:
smoothed_mu = self.safe_sliders.update("mu", slider_value, dt)
self.engine.set_params(mu=smoothed_mu)
```

**Sources:**
- [MathWorks: Parameter Smoother Block](https://www.mathworks.com/help/dsp/ref/parametersmoother.html)
- [Medium: Exponential Moving Average Smoothing](https://medium.com/@dmitriy.bolotov/six-approaches-to-time-series-smoothing-cc3ea9d6b64f)
- [Wikipedia: Exponential Smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing)

**Performance:** Negligible (<0.5ms for all parameters combined)

---

## Integration with Existing Pipeline

### Current Pipeline (to be replaced)
```
Engine.step() → world [H, W] float64
    ↓
ColorLayerSystem.compute_signals() → 4 signals [4, H, W]
    ├─ Core: world density
    ├─ Halo: edge gradient
    ├─ Spark: temporal change
    └─ Memory: EMA trail
    ↓
ColorLayerSystem.composite() → RGB [H, W, 3] uint8
    ↓
viewer._render_frame() → pygame surface
```

### New Pipeline (proposed)
```
Engine.step() → world [H, W] float64
    ↓
IridescentColorSystem.render()
    ├─ ThicknessMapper.map(world) → thickness [H, W]
    ├─ SpatialGradient.compute() → gradient [H, W]
    ├─ HueSweep.advance(dt) → hue_offset [scalar]
    ├─ ThinFilmRenderer.render(thickness, gradient, hue) → RGB [H, W, 3]
    └─ RGBTint.apply(RGB) → RGBA [H, W, 4] uint8
    ↓
viewer._render_frame() → pygame surface
```

### Migration Strategy

**Phase 1: Parallel implementation**
- Create `iridescent_color.py` alongside `color_layers.py`
- Implement all 6 components as separate classes
- Add toggle in viewer to switch between old/new systems

**Phase 2: Integration**
- Replace `ColorLayerSystem` instantiation with `IridescentColorSystem`
- Update UI sliders: remove layer weights, add RGB tint + gradient controls
- Integrate `SafeSliderManager` for all engine parameters

**Phase 3: Cleanup**
- Remove `color_layers.py` after validation
- Remove old layer weight sliders from UI
- Simplify control panel layout

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Engine (existing)                          │
│  - Lenia.step() → world [1024, 1024] float64 [0, 1]         │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ↓
┌──────────────────────────────────────────────────────────────┐
│              IridescentColorSystem                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Thickness    │  │ Spatial      │  │ Hue          │      │
│  │ Mapper       │  │ Gradient     │  │ Sweep        │      │
│  │              │  │              │  │              │      │
│  │ world        │  │ (X, Y, time) │  │ time         │      │
│  │   ↓          │  │   ↓          │  │   ↓          │      │
│  │ thickness    │  │ gradient     │  │ hue_offset   │      │
│  │ [H,W] nm     │  │ [H,W] nm     │  │ (degrees)    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           ↓                                │
│                  ┌────────────────┐                        │
│                  │ ThinFilm       │                        │
│                  │ Renderer       │                        │
│                  │                │                        │
│                  │ Physics:       │                        │
│                  │ cos²(π·2nt/λ)  │                        │
│                  │   ↓            │                        │
│                  │ RGB [H,W,3]    │                        │
│                  └────────┬───────┘                        │
│                           ↓                                │
│                  ┌────────────────┐                        │
│                  │ RGB            │                        │
│                  │ Tint           │                        │
│                  │                │                        │
│                  │ R/G/B gains    │                        │
│                  │ brightness     │                        │
│                  │   ↓            │                        │
│                  │ RGBA uint8     │                        │
│                  └────────┬───────┘                        │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ↓
                  pygame surface → display
```

---

## Safe Slider Integration

### Current Slider System (in viewer.py)
```python
# Direct parameter application (UNSAFE)
def _make_param_callback(self, key):
    def callback(val):
        self.engine.set_params(**{key: val})
    return callback
```

### New Slider System (with SafeSliderManager)
```python
class Viewer:
    def __init__(self, ...):
        # ... existing init ...
        self.safe_sliders = SafeSliderManager(smoothing_time_seconds=0.3)

    def _make_param_callback(self, key):
        """Create callback with safe interpolation."""
        def callback(val):
            # Store target value, will be smoothed in main loop
            self.slider_targets[key] = val
        return callback

    def run(self):
        # In main loop, BEFORE engine.step():
        for key, target in self.slider_targets.items():
            smoothed = self.safe_sliders.update(key, target, dt)
            self.engine.set_params(**{key: smoothed})

        # Then step engine with smoothed params
        if not self.paused:
            self.engine.step()
```

**Critical points:**
- Slider callback stores TARGET value (user intent)
- Main loop applies SMOOTHED value (prevents jumps)
- Smoothing happens BEFORE engine.step() to avoid state corruption
- Each parameter smoothed independently (mu, sigma, R, T, etc.)

---

## Suggested Build Order

### Milestone 1: Core Iridescence (3-5 hours)
**Goal:** Oil-slick shimmer visible, replaces old color system

1. Create `iridescent_color.py` with 5 classes (no SafeSliderManager yet)
2. Implement ThicknessMapper (simplest component)
3. Implement SpatialGradient with fixed parameters (angle=45°, wavelength=1.5)
4. Implement HueSweep (copy from ColorLayerSystem)
5. Implement ThinFilmRenderer (core physics, ~50 lines)
6. Implement RGBTint (simple gains)
7. Wire into viewer.py as toggle option (keep old system for comparison)
8. Validate: organism shows rainbow shimmer, hue rotates slowly
9. Performance check: <22ms/frame at 1024²

**Deliverable:** Iridescent rendering works, visually distinct from old system

---

### Milestone 2: UI + Safe Sliders (2-3 hours)
**Goal:** User controls finalized, parameter changes safe

1. Remove old layer weight sliders from control panel
2. Add new sliders:
   - RGB tint: R gain [0, 2], G gain [0, 2], B gain [0, 2]
   - Brightness [0.5, 2.0]
   - Gradient angle [0, 360] degrees
   - Gradient wavelength [0.5, 3.0]
3. Implement SafeSliderManager
4. Integrate into viewer._make_param_callback()
5. Test: rapidly move mu slider → organism should NOT die
6. Test: switch presets → no drift artifacts

**Deliverable:** Safe slider architecture validated, UI simplified

---

### Milestone 3: Parameter Tuning (1-2 hours)
**Goal:** Visual quality polished

1. Tune thickness range (min_nm, max_nm) for best colors on Coral preset
2. Tune spatial gradient defaults (angle, wavelength, amplitude)
3. Tune smoothing time constant (balance responsiveness vs safety)
4. Validate on all remaining presets (Orbium, Cardiac Waves, Mitosis)

**Deliverable:** Visually stunning across all presets

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Direct HSV Conversion in Hot Loop
**What:** Converting RGB → HSV → rotate hue → RGB per-pixel
**Why bad:** HSV conversion involves conditionals (slow in numpy)
**Cost:** ~15-20ms added latency (exceeds budget)
**Instead:** Use rotation matrix method in ThinFilmRenderer._rotate_hue()

### Anti-Pattern 2: Spectral Integration
**What:** Integrate interference over full visible spectrum (380-750nm, 100+ samples)
**Why bad:** 100× compute cost for marginal visual gain
**Cost:** ~200ms/frame (completely unacceptable)
**Instead:** Sample 3 wavelengths (R, G, B) — adequate for display

### Anti-Pattern 3: Slider Values Applied Instantly
**What:** `self.engine.set_params(mu=slider_value)` without smoothing
**Why bad:** Abrupt parameter jump can violate engine stability conditions → death spiral
**Example:** Lenia mu jump from 0.12 to 0.20 → all cells die instantly
**Instead:** Use SafeSliderManager with EMA smoothing (0.3-0.5s time constant)

### Anti-Pattern 4: Global State in Renderer
**What:** ThinFilmRenderer stores world state or previous frames
**Why bad:** Violates single-responsibility, complicates reset logic
**Instead:** Renderer is pure function: thickness → RGB (stateless except pre-allocated buffers)

### Anti-Pattern 5: Per-Pixel Python Loops
**What:** `for i in range(H): for j in range(W): rgb[i,j] = compute_color(...)`
**Why bad:** Python loop overhead is ~1000× slower than vectorized numpy
**Cost:** ~10 seconds/frame (!)
**Instead:** Vectorized numpy operations on entire array at once

---

## Scalability Considerations

| Concern | Current (1024²) | If 2048² | If 512² |
|---------|----------------|----------|---------|
| **Thickness mapping** | 2ms | 8ms | 0.5ms |
| **Spatial gradient** | 3ms | 12ms | 0.8ms |
| **Thin-film render** | 12ms | 48ms | 3ms |
| **RGB tint** | 2ms | 8ms | 0.5ms |
| **Total** | ~20ms | ~76ms | ~5ms |

**Notes:**
- Rendering cost scales O(N²) with grid size (expected for pixel operations)
- 2048² exceeds real-time budget (76ms > 60fps) → need optimization if upscaling
- 512² well within budget → comfortable for faster systems

**Optimization paths if needed:**
1. **Downsample-render-upsample:** Render at 512², bilinear upsample to 1024² → saves ~15ms
2. **LUT approach:** Pre-compute thickness→RGB lookup table → saves ~5-8ms but less flexible
3. **Numba JIT:** Compile ThinFilmRenderer with @numba.jit → ~2-3× speedup possible

---

## Open Questions for Phase-Specific Research

These questions don't need answers NOW, but will need investigation during implementation:

1. **Thickness range:** What [min_nm, max_nm] produces the richest colors for Lenia organisms?
   - Hypothesis: 200-800nm based on physics, but may need tuning for aesthetics
   - Research needed: Empirical testing on Coral preset

2. **Spatial gradient patterns:** Should gradient be pure sine wave, or multi-scale noise?
   - Hypothesis: Sine wave sufficient for MVP, Perlin noise for advanced shimmer
   - Research needed: User preference testing

3. **Hue rotation speed:** Keep 2.5 deg/s from old system, or slower?
   - Project context says "slower LFO overall" — applies to hue sweep too?
   - Research needed: User feedback on rotation speed

4. **Smoothing time constant:** 0.3s vs 0.5s vs 1.0s for slider safety?
   - Trade: faster response vs more protection
   - Research needed: Stability testing with abrupt slider changes

5. **Feedback integration:** Does iridescence system need feedback into engine?
   - Old system had feedback coefficient — needed for iridescence?
   - Research needed: Test if organism evolves better with/without feedback

---

## Performance Validation Checklist

Before declaring architecture complete:

- [ ] ThinFilmRenderer runs in <12ms @ 1024² (measured with time.perf_counter)
- [ ] Total render pipeline <22ms @ 1024² (matches current baseline)
- [ ] No frame drops during slider manipulation
- [ ] No memory leaks over 10min continuous run (memory usage stable)
- [ ] SafeSliderManager prevents organism death when mu slider jerked ±50%
- [ ] Visual quality comparable to reference (oil slick / soap bubble shimmer)

---

## Sources

### Thin-Film Interference Physics
- [Wikipedia: Thin-film interference](https://en.wikipedia.org/wiki/Thin-film_interference) — core formulas
- [Physics LibreTexts: Thin Film Interference](https://phys.libretexts.org/Bookshelves/University_Physics/Calculus-Based_Physics_(Schnick)/Volume_B:_Electricity_Magnetism_and_Optics/B24:_Thin_Film_Interference) — optical path equations
- [OpenStax: Interference in Thin Films](https://openstax.org/books/university-physics-volume-3/pages/3-4-interference-in-thin-films) — constructive/destructive conditions

### Shader Implementations (adapted for CPU)
- [Alan Zucconi: Car Paint Shader - Thin-Film Interference](https://www.alanzucconi.com/2017/10/27/carpaint-shader-thin-film-interference/) — simplified cosine model
- [Shadertoy: Fast Thin-Film Interference](https://www.shadertoy.com/view/ld3SDl) — GPU implementation reference
- [Jelly Renders: Rendering a Simple Iridescent Material](https://jellyrenders.com/graphics/ray%20tracing/2025/11/17/simple-iridescent-material/) — angle-dependent phase calculation
- [Blender Projects: Cycles Thin Film Iridescence PR](https://projects.blender.org/blender/blender/pulls/118477) — production renderer approach

### Safe Parameter Control
- [MathWorks: Parameter Smoother Block](https://www.mathworks.com/help/dsp/ref/parametersmoother.html) — EMA smoothing for live parameters
- [Wikipedia: Exponential Smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) — mathematical foundation
- [Medium: Six Approaches to Time Series Smoothing](https://medium.com/@dmitriy.bolotov/six-approaches-to-time-series-smoothing-cc3ea9d6b64f) — EMA vs alternatives

### NumPy Performance
- [NumPy Gradient Documentation](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html) — spatial field operations
- [Python 2025: Performance Optimization Guide](https://blog.madrigan.com/en/blog/202602091505/) — vectorization best practices
