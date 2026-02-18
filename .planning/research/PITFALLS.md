# Integration Pitfalls: Scope Plugin + CuPy GPU + RunPod Deployment

**Domain:** Wrapping existing pygame/numpy cellular automata as a DayDream Scope text-only plugin, porting to CuPy for GPU acceleration, deploying alongside a 14B model on RunPod

**Researched:** 2026-02-17
**Confidence:** MEDIUM-HIGH (based on direct code audit of existing codebase + known patterns from Scope plugin template, CuPy documentation, and GPU memory constraints)

---

## Critical Pitfalls

Mistakes that cause rewrites, black frames, OOM crashes, or the plugin failing to load entirely.

---

### Pitfall 1: Storing State in `__init__()` That Belongs in `__call__()`

**What goes wrong:** The entire `Viewer` class stores all runtime state (engine, LFO systems, preset, flow params, color pipeline) in `__init__()`. If you lift this logic naively into a Scope pipeline class, runtime parameters (preset, speed, hue, flow strengths) get initialized once at load time and never update when the user moves a Scope UI slider.

**Why it happens:** The Scope plugin contract is explicit: runtime parameters MUST be read from `kwargs` in `__call__()`, not stored in `__init__()`. The CLAUDE.md and example plugin both say this. But the existing `Viewer` class does the opposite — it stores everything as instance attributes set at init time.

**Specific code at risk in this codebase:**
- `viewer.py Viewer.__init__()` sets `self.sim_speed`, `self._flow_radial`, `self._flow_rotate` (and all other flow keys) as instance attributes
- `_apply_preset()` is called in `__init__()` to set engine state
- LFO systems are created in `__init__()` with preset values

**Consequences:**
- Scope UI sliders do nothing — parameters never update
- Changing preset in Scope UI has no effect
- Plugin works on first frame, then freezes at initial state

**Prevention:**
1. **Read ALL user-controllable params from kwargs on every call:**
   ```python
   def __call__(self, prompt: str, **kwargs):
       preset = kwargs.get("preset", "coral")
       speed = kwargs.get("speed", 1.0)
       hue = kwargs.get("hue", 0.25)
       flow_rotate = kwargs.get("flow_rotate", 0.5)
   ```
2. **Initialize simulation state once in `__init__()`, params never.** Engine (`Lenia`, etc.), color pipeline (`IridescentPipeline`), buffers are load-time. Everything the user controls is runtime from kwargs.
3. **Preset change requires engine reinit — handle it:** If `kwargs["preset"]` differs from `self._current_preset`, call `_apply_preset()` in `__call__()`. Track current preset as instance state to detect changes.
4. **Test by moving each slider.** If moving a Scope slider has no effect on output, that param is still stored in `__init__()`.

**Phase:** Plugin Wrapper (Phase 2) — this is the architectural foundation of the wrapper.

---

### Pitfall 2: Importing pygame at Module Level in the Plugin Package

**What goes wrong:** `viewer.py` imports pygame at the top. When Scope loads the plugin package, Python imports the package `__init__.py` and all referenced modules. If pygame is imported at module level anywhere in the package, the import fails on RunPod (no display server, no SDL).

**Why it happens:** pygame (and pygame-ce) require SDL, which tries to connect to a display server on import. On a headless RunPod container, there is no X11 or Wayland server. The import raises `pygame.error: No available video device` or `SDL_Init failed`. This kills the plugin load before `register_pipelines()` is ever called.

**Specific risk in this codebase:**
- `viewer.py` has `import pygame` at line 4 (top level)
- `controls.py` almost certainly imports pygame (it defines `ControlPanel`)
- `__main__.py` likely imports `Viewer` which imports pygame
- The plugin's `__init__.py` may import from `viewer.py`

**Consequences:**
- Plugin fails to load with a cryptic SDL error
- `register_pipelines()` never runs
- No visible Scope error — plugin just doesn't appear in the pipeline list

**Prevention:**
1. **Create a clean `pipeline.py` for Scope that imports ONLY the engine files**, not `viewer.py` or `controls.py`:
   ```python
   # scope_pipeline.py — NO pygame imports
   from .lenia import Lenia
   from .smoothlife import SmoothLife
   from .mnca import MNCA
   from .gray_scott import GrayScott
   from .iridescent import IridescentPipeline
   from .presets import get_preset
   # Do NOT import viewer.py, controls.py, smoothing.py (if it has pygame)
   ```
2. **Audit every file the plugin imports for `import pygame`.** `viewer.py` and `controls.py` are known offenders. `smoothing.py` is likely clean.
3. **Keep `viewer.py` standalone** — it stays as the interactive viewer, the plugin does not import it.
4. **Set `SDL_VIDEODRIVER=dummy` as a fallback** in the plugin's `__init__` if pygame must be imported for some reason:
   ```python
   import os
   os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
   import pygame
   pygame.init()
   ```
   But this is a last resort — better to avoid the pygame import entirely.

**Phase:** Plugin Wrapper (Phase 2) — must get this right from the start.

---

### Pitfall 3: CuPy `rfft2()` Producing Different Float Precision Than NumPy

**What goes wrong:** After porting `np.fft.rfft2()` to `cupy.fft.rfft2()`, the Lenia and SmoothLife simulations produce subtly different patterns. The organisms look different from the CPU version, or they die at different times, or the threshold behaviors (MNCA rules, GS dead zone) fire at different values.

**Why it happens:** CuPy uses cuFFT under the hood, which uses different floating-point rounding paths than NumPy's pocketfft. For float32 FFT at 1024x1024:
- CuPy rfft2 operates entirely in float32 (faster but less precise)
- NumPy rfft2 internally upgrades to float64 for accumulation, then rounds back
- The difference is typically 1e-6 to 1e-4 per element
- After many iterations, small deltas compound: Lenia mu=0.15 is in a narrow viability window; a 1e-4 shift in the convolution output can shift the growth function enough to change organism behavior

**Specific risk in this codebase:**
- `lenia.py`: `U = np.fft.irfft2(world_fft * self._kernel_fft, s=(self.size, self.size))` — result feeds directly into `self._bell(U, self.mu, self.sigma)` with very narrow tolerance (sigma=0.017)
- `smoothlife.py`: sigmoid computations `1/(1+exp(...))` are sensitive to small input differences
- `mnca.py`: boolean threshold rules `avg >= low` can flip on numerical noise

**Consequences:**
- Visually different organisms (not always worse, just different)
- MNCA threshold rules fire differently — organism may die faster or grow differently
- GS dead zone (V<0.015) may miss pixels that NumPy would catch, breaking the clean black background
- SmoothLife contrast curve ^1.4 amplifies any precision differences

**Prevention:**
1. **Accept that outputs will differ, but verify organism survives.** The goal is not bit-identical output, it is "organism continues evolving and looks good."
2. **Run both CPU and GPU versions for 500+ steps on the same preset and compare visually.** If GPU version dies where CPU version lives, precision is the culprit.
3. **Force float32 explicitly everywhere** — don't let CuPy silently upgrade:
   ```python
   world_fft = cp.fft.rfft2(self.world.astype(cp.float32))
   ```
4. **Test threshold-sensitive presets first:** MNCA (boolean rules), GS (dead zone V<0.015). These are most sensitive to precision.
5. **The SmoothLife contrast curve `^1.4` is applied in the viewer, not the engine** — verify that the viewer-side rendering still produces the correct curve after GPU port.
6. **Do NOT mix cupy arrays and numpy arrays in a single computation** — implicit conversion is silently slow and breaks the float32 guarantee.

**Phase:** CuPy Port (Phase 3) — must validate all 4 engines before declaring success.

---

### Pitfall 4: VRAM OOM When CA Plugin and 14B Model Share the Same GPU

**What goes wrong:** The CA plugin allocates GPU arrays, runs fine alone, then crashes with CUDA out-of-memory when Krea Realtime loads the 14B WAN model alongside it.

**Why it happens:**
- Wan2.1-T2V-14B in float16 requires ~28GB VRAM minimum at inference
- RTX 5090 has 32GB VRAM — leaving only ~4GB for CA plugin + CUDA overhead
- CuPy arrays for 1024x1024 float32: `world` (4MB), 2+ kernel FFTs (complex64, ~8MB each), color pipeline buffers (3 channels × float32 ~12MB), noise pool (6 frames × 4MB = 24MB), flow fields (14 arrays × 4MB = 56MB). Total: ~120MB

**The math:**
- 14B model: ~28,000MB
- CUDA contexts + PyTorch + model overhead: ~1,000MB
- CA plugin GPU: ~120MB (minimal, but it's not zero)
- Available headroom: ~2,880MB — thin
- Activation memory during Krea inference: 1-3GB spikes
- Risk of OOM during inference spike: HIGH

**Additional risks:**
- If CA plugin pre-allocates a noise pool on GPU (`cp.random.randn(1024, 1024, dtype=cp.float32)` × 6), this is 24MB permanently held during model inference
- CuPy's memory pool (`cp.get_default_memory_pool()`) holds freed memory for reuse — may not release fast enough when model needs it

**Consequences:**
- `torch.cuda.OutOfMemoryError` during Krea inference
- Plugin crashes silently, Scope shows black frames
- RunPod pod requires restart

**Prevention:**
1. **Drop to 512x512 for GPU mode by default.** 1024x1024 was needed for CPU visual quality; GPU can deliver equal quality at 512 because it processes more steps per second. Saves 4x VRAM: 512×512 noise pool = 6MB instead of 24MB.
2. **Explicitly limit CuPy memory pool:**
   ```python
   import cupy as cp
   pool = cp.get_default_memory_pool()
   pool.set_limit(size=512 * 1024 * 1024)  # 512MB max for CA plugin
   ```
3. **Keep flow fields on CPU, transfer per-call if needed.** Flow fields are 14 arrays of 4MB each (56MB). They're pre-computed constants — keep as numpy, convert to cupy only for the advection step:
   ```python
   vx_gpu = cp.asarray(vx_cpu)  # transfer only active flows
   ```
4. **Measure actual VRAM usage before and after Krea model load:**
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv
   ```
5. **Test on RunPod with 14B model loaded first**, then load CA plugin, not the other way around. If it OOMs loading second, you need to reduce CA GPU footprint.
6. **Consider CPU-side rendering pipeline.** The `IridescentPipeline` color math (LUT lookups, numpy take) is fast on CPU. Only port the FFT-heavy simulation engines to GPU, keep rendering on CPU.

**Phase:** RunPod Deployment (Phase 4) — but VRAM budget must be designed in Phase 3 (CuPy).

---

### Pitfall 5: GS (Gray-Scott) Runs at 512x512 but CuPy Kernel Is Built for Current Size

**What goes wrong:** The existing code dynamically switches `sim_size` between 512 (GS) and 1024 (others) in `_apply_preset()`. In the Scope plugin, the engine is initialized once in `__init__()`. If the preset changes from a 1024 preset to a GS preset, the engine is still sized at 1024, or vice versa. The pre-allocated buffers in `gray_scott.py` (`self._padded`, `self._lap_U`, `self._lap_V`) are the wrong shape.

**Why it happens:** `gray_scott.py` pre-allocates `self._padded = np.zeros((size + 2, size + 2))` at init. If you create a GS engine at 1024 (wrong) or a Lenia engine at 512 (too small), computations on wrong-shaped arrays cause errors or wrong results.

**Specific risk:** When porting to CuPy, all these pre-allocated buffers become CuPy arrays. Rebuilding a 1024x1024 CuPy FFT kernel takes ~200ms the first time. If the plugin switches engines frequently (preset change → engine change → kernel rebuild), the first frame after a switch has a 200ms stall.

**Consequences:**
- Dropped frames or stall on preset switch
- Possible `ValueError: operands could not be broadcast` if buffer shapes don't match
- GS at 1024x1024 is 4x slower than 512x512, may not sustain real-time

**Prevention:**
1. **Match GS to 512x512, all others to a consistent size.** Encode this as a constant in the plugin, not a dynamic decision:
   ```python
   ENGINE_SIZES = {"gray_scott": 512, "lenia": 512, "smoothlife": 512, "mnca": 512}
   # Use 512 on GPU everywhere — GPU makes 512 fast enough
   ```
2. **Pre-build all engines at init, swap on preset change.** Don't destroy and recreate — keep all 4 engine instances alive in `__init__()`, switch the active pointer in `__call__()` when preset changes.
3. **Cache CuPy kernel FFTs.** On first use per preset, build FFT and cache it. Subsequent calls to same preset skip the build:
   ```python
   self._kernel_fft_cache = {}
   ```
4. **Pre-warm all engines at plugin init.** Run 200 steps on each engine at startup so first-frame stalls don't happen at runtime.

**Phase:** Plugin Wrapper (Phase 2) and CuPy Port (Phase 3).

---

### Pitfall 6: `scipy.ndimage.map_coordinates` Has No Direct CuPy Equivalent

**What goes wrong:** The flow field advection in `viewer.py:_advect()` calls `scipy.ndimage.map_coordinates(field, [adv_y, adv_x], order=1, mode='wrap')`. When porting to CuPy, you reach for `cupyx.scipy.ndimage.map_coordinates()` — it exists, but with restrictions that break the existing call pattern.

**The actual situation (HIGH confidence from CuPy docs):**
- `cupyx.scipy.ndimage.map_coordinates` exists as of CuPy 9+
- Restriction: `mode='wrap'` MAY not be supported depending on CuPy version. The supported modes have changed across versions.
- The `order=1` (bilinear) parameter is supported
- Performance: First call compiles a CUDA kernel; subsequent calls are fast but the first-call compile may take 3-5 seconds

**Specific risk in this codebase:**
```python
# viewer.py line ~709
return _map_coordinates(
    field, [self._flow_adv_y, self._flow_adv_x],
    order=1, mode='wrap'
).astype(np.float32)
```
The `mode='wrap'` is critical — CA simulations use periodic/wrap boundary conditions.

**Consequences:**
- If `mode='wrap'` not supported: edges get reflected or clamped, creating visual artifacts at boundaries
- Organisms creep to one side instead of wrapping
- Compile time on first call may cause a multi-second stall

**Prevention:**
1. **Verify CuPy version on RunPod first:**
   ```bash
   python -c "import cupy; print(cupy.__version__)"
   python -c "from cupyx.scipy.ndimage import map_coordinates; help(map_coordinates)"
   ```
2. **Implement a fallback bilinear warp using pure CuPy:**
   ```python
   def _cupy_bilinear_wrap(field, coords_y, coords_x):
       H, W = field.shape
       y = cp.mod(coords_y, H).astype(cp.int32)
       x = cp.mod(coords_x, W).astype(cp.int32)
       # Bilinear: sample 4 corners
       y1 = (y + 1) % H
       x1 = (x + 1) % W
       fy = cp.mod(coords_y, 1.0)  # fractional part
       fx = cp.mod(coords_x, 1.0)
       return (field[y, x] * (1-fy) * (1-fx) +
               field[y1, x] * fy * (1-fx) +
               field[y, x1] * (1-fy) * fx +
               field[y1, x1] * fy * fx)
   ```
   This is equivalent to `map_coordinates(order=1, mode='wrap')` with no dependency on cupyx.
3. **Test wrap behavior explicitly.** Seed an organism near the edge — it should "wrap around" not stop at the boundary.
4. **Pre-trigger the CUDA kernel compile at init** by calling `map_coordinates` once with dummy data to eliminate first-call stall.

**Phase:** CuPy Port (Phase 3) — investigate before committing to cupyx dependency.

---

### Pitfall 7: THWC Tensor Shape and [0,1] Range Requirements

**What goes wrong:** The plugin returns the wrong tensor shape or value range. Scope expects `THWC` (Time, Height, Width, Channels) with values in `[0, 1]` as a `torch.Tensor`. Common mistakes: returning HWC (missing time dimension), returning uint8 [0,255], returning a numpy array instead of torch tensor, or returning float32 on CPU when Scope expects GPU tensor.

**Specific risk in this codebase:**
- `IridescentPipeline.render()` returns `np.ndarray` of shape `(H, W, 3)` as `uint8` in `[0, 255]`
- The plugin must convert: `(H, W, 3) uint8` → `(1, H, W, 3) float32 [0,1]` torch tensor
- Current output is `display_buffer` — a numpy uint8 array, NOT a torch tensor, NOT THWC

**The required conversion:**
```python
rgb_uint8 = self.iridescent.render(...)  # (H, W, 3) uint8
frame = torch.from_numpy(rgb_uint8.copy()).float() / 255.0  # (H, W, 3) float32
frame = frame.unsqueeze(0)  # (1, H, W, 3) THWC
return frame
```

**Consequences:**
- Scope receives wrong shape → error or scrambled video
- Returning uint8 → Scope clamps to [0,1] → black frame
- Missing `.copy()` on numpy array → torch tensor backed by numpy memory → undefined behavior if numpy array is recycled next frame

**Prevention:**
1. **Always unsqueeze(0) to add T=1 dimension.** Text-only pipelines produce one frame per call.
2. **Always divide by 255.0 if returning uint8 source.** The `display_buffer` in `IridescentPipeline` is uint8.
3. **Always `.copy()` before `torch.from_numpy()`** — if the underlying numpy buffer is reused (pre-allocated), the tensor will see next frame's data.
4. **Verify output shape in plugin `__call__` with an assertion during development:**
   ```python
   assert output.shape == (1, H, W, 3), f"Wrong shape: {output.shape}"
   assert output.dtype == torch.float32
   assert output.min() >= 0.0 and output.max() <= 1.0
   ```
5. **For CuPy path:** Convert cupy array to numpy before torch: `cp.asnumpy(gpu_rgb)` then the normal numpy→torch path.

**Phase:** Plugin Wrapper (Phase 2) — get this right on the first frame.

---

## Moderate Pitfalls

---

### Pitfall 8: Headless Warmup — GS Needs 1000 Steps, Blocks `__init__()`

**What goes wrong:** The existing `_apply_preset()` runs 1000 warmup steps for GS and 200 steps for Lenia in `__init__()`. On CPU, 1000 GS steps at 512×512 takes ~15-20 seconds. This blocks the plugin from loading. Scope likely has a timeout for plugin initialization and may kill the plugin.

**Why it happens:** Warmup is needed so the user sees developed structure immediately (not a blank or random seed). On CPU it's slow. Even on GPU at 512×512, 1000 GS steps with CuPy may take 3-5 seconds on first run due to kernel compilation overhead.

**Consequences:**
- Plugin load timeout → plugin not available
- If no timeout: Scope UI freezes during warmup
- User sees blank frame for 15+ seconds after switching to GS preset

**Prevention:**
1. **Move warmup to first `__call__()`, not `__init__()`.** Track `_is_warmed_up = False` per engine. On first call, run warmup steps, then return the frame. The first frame may be slow but plugin loads instantly.
2. **Reduce warmup steps for GPU.** GPU can run ~10× more steps per second, so 100 GS steps on GPU ≈ 1000 on CPU visually. Tune down.
3. **Use a background thread for warmup.** Initialize engine in thread, return blank frames until warmup is done:
   ```python
   import threading
   self._warmup_done = threading.Event()
   threading.Thread(target=self._run_warmup, daemon=True).start()
   ```
4. **GS at 512×512 on RTX 5090 is fast.** Profile before assuming warmup time is a problem on GPU.

**Phase:** Plugin Wrapper (Phase 2).

---

### Pitfall 9: `scipy.ndimage.gaussian_filter` and `scipy.ndimage.zoom` Not in CuPy

**What goes wrong:** `viewer.py` uses `scipy.ndimage.gaussian_filter` and `scipy.ndimage.zoom` for dilation, color offset building, and world blurring. When porting to CuPy, these exist in `cupyx.scipy.ndimage` — but `zoom` does NOT exist in cupyx. The `gaussian_filter` exists but with more limited parameter support.

**Specific risk:**
- `_dilate_world()` uses `_scipy_maximum`, `_scipy_gaussian`, `_scipy_zoom`
- `_blur_world()` uses `_scipy_gaussian` and `_scipy_zoom`
- `_build_color_offset()` uses `_scipy_gaussian` and `_scipy_zoom`
- These are called at init (for building static fields like `_color_offset`) and potentially per-frame (dilation)

**The good news:** `_build_color_offset()`, `_containment`, `_mnca_containment`, `_noise_pool`, flow fields are ALL pre-computed at init. They can be built on CPU and transferred to GPU once. Only per-frame computation needs GPU.

**Prevention:**
1. **Build all static fields on CPU, transfer to GPU once:**
   ```python
   # __init__(): build on CPU
   self._color_offset_cpu = self._build_color_offset(size)
   self._containment_cpu = self._build_containment(size)
   # After cupy import confirmed:
   self._color_offset_gpu = cp.asarray(self._color_offset_cpu)
   ```
2. **For per-frame dilation (`_dilate_world`): implement without zoom:**
   ```python
   def _dilate_world_gpu(world, thickness):
       if thickness < 0.5:
           return world
       factor = 4
       small = world[::factor, ::factor]
       if thickness >= 4:
           kernel_size = max(2, int(thickness / factor))
           small = cupyx.scipy.ndimage.maximum_filter(small, size=kernel_size)
       small = cupyx.scipy.ndimage.gaussian_filter(small, sigma=max(0.5, thickness/factor))
       # Bilinear upsample without zoom: use cp.kron or repeat + gaussian
       return cp.repeat(cp.repeat(small, factor, axis=0), factor, axis=1)[:H, :W]
   ```
3. **The `world blur` in viewer.py is also affected** — same approach: implement bilinear upsampling in CuPy without zoom.
4. **Do not block plugin init on cupy compilation.** The first call to each cupyx function compiles a CUDA kernel. Pre-trigger with dummy data at init.

**Phase:** CuPy Port (Phase 3).

---

### Pitfall 10: Scope Calls `__call__()` at Video Frame Rate — Not at Sim Frame Rate

**What goes wrong:** Scope calls the pipeline's `__call__()` at the video output rate (likely 24-30 fps). The CA simulation at CPU speed runs at 12-55 fps depending on engine. At GPU speed, it could run at 100+ fps. If `__call__()` runs 1 sim step per call, the organism evolves too fast relative to video output, or too slow if the sim is rate-limited.

**Why it happens:** The existing `Viewer` uses a `speed_accumulator` with fractional step advancement:
```python
self.speed_accumulator += self.sim_speed * dt
while self.speed_accumulator >= 1.0:
    self.engine.step()
    self.speed_accumulator -= 1.0
```
This decouples sim rate from frame rate. In the Scope plugin, there is no `dt` (real elapsed time). `__call__()` may be called irregularly.

**Consequences:**
- Without rate management: organism evolves at display frame rate (constant)
- If GPU is fast, the organism may appear to "sprint" through its state space too quickly, losing the slow organic feel
- If sim is slow, Scope may drop frames waiting for `__call__()` to return

**Prevention:**
1. **Track real time in `__call__()` using `time.perf_counter()`:**
   ```python
   now = time.perf_counter()
   dt = now - self._last_call_time
   self._last_call_time = now
   ```
2. **Implement the same `speed_accumulator` pattern inside `__call__()`.**
3. **Read speed from kwargs:** `speed = kwargs.get("speed", 1.0)` — user-controllable.
4. **Cap max steps per call** to prevent runaway if Scope calls infrequently:
   ```python
   max_steps_per_call = 5
   steps = min(int(steps_to_take), max_steps_per_call)
   ```

**Phase:** Plugin Wrapper (Phase 2).

---

### Pitfall 11: CuPy Import Fails Silently, Plugin Falls Back to CPU Without Warning

**What goes wrong:** On a RunPod pod that doesn't have CuPy installed (or has the wrong CUDA toolkit version), `import cupy` fails. If the plugin catches this silently and falls back to numpy, the user sees the plugin running but doesn't know it's on CPU. On 14B inference pods, this means the CA is eating CPU cycles that should be idle.

**The actual CuPy/CUDA compatibility matrix:**
- CuPy version must match the installed CUDA toolkit version exactly
- RTX 5090 requires CUDA 12.x and CuPy built for CUDA 12.x
- If Scope's container has CUDA 11.x and you install CuPy 12.x, import fails
- The `daydreamlive/scope:main` container may or may not have CuPy pre-installed

**Prevention:**
1. **Check CuPy availability at plugin init, log clearly:**
   ```python
   try:
       import cupy as cp
       cp.array([1])  # Force actual GPU operation to test
       CUPY_AVAILABLE = True
       print("[CA Plugin] CuPy available — GPU acceleration active")
   except Exception as e:
       CUPY_AVAILABLE = False
       print(f"[CA Plugin] CuPy unavailable ({e}) — falling back to NumPy")
   ```
2. **Check CUDA toolkit version before installing CuPy on RunPod:**
   ```bash
   nvcc --version  # or nvidia-smi
   # Install matching CuPy: pip install cupy-cuda12x for CUDA 12.x
   ```
3. **Do not silently fall back** — log the fallback prominently. Silent CPU fallback on a 32GB VRAM pod is a wasted opportunity.
4. **Test the numpy fallback path.** The plugin must work correctly on CPU even if CuPy is unavailable (for local development).

**Phase:** RunPod Deployment (Phase 4).

---

### Pitfall 12: GS Dead Zone Breaks After CuPy Port

**What goes wrong:** Gray-Scott's visual relies on `V < 0.015 → black background`. In NumPy, this is straightforward. In CuPy, if there's a float32 precision difference in the V channel, some pixels that should be V=0.013 (black) end up as V=0.016 (visible), causing a grainy gray background instead of pure black.

**Why it happens:** The GS laplacian uses a pre-allocated `_padded` buffer with manual boundary assignment. In the CuPy port, if `_padded` is a cupy array and the slice assignments use different rounding, the laplacian accumulates slightly differently over 1000 steps.

**The rendering path at risk:**
```python
# viewer.py _render_gs_emboss():
V_stretch = np.clip(V * 4.0, 0.0, 1.0)
# Dead zone: V < 0.015 → stays near 0 → black
# Contrast: V * 4.0 means V=0.015 → 0.06 → just above black
```

**Prevention:**
1. **Explicitly apply dead zone in the rendering step, not just relying on contrast:**
   ```python
   dead_zone_mask = V < 0.015
   V_stretch = cp.clip(V * 4.0, 0.0, 1.0)
   V_stretch[dead_zone_mask] = 0.0  # Explicit kill
   ```
2. **Verify dead zone after 1000 warmup steps.** If mean of V array near edges is > 0.01, the containment feed mask may not be working correctly on GPU.
3. **The GS `_padded` boundary conditions** are critical — the `p[0, 0] = field[-1, -1]` corner assignments must be correct on GPU. CuPy handles numpy-style indexing identically, but verify.

**Phase:** CuPy Port (Phase 3) — GS is the most visually sensitive engine.

---

### Pitfall 13: `pyproject.toml` Entry Point Must Match Package Structure Exactly

**What goes wrong:** The Scope plugin system discovers plugins via Python entry points. If `pyproject.toml` declares:
```toml
[project.entry-points."scope"]
cellular_automata = "cellular_automata.plugin"
```
But the package is installed as `cellular_automata_plugin` or the module path is wrong, the plugin never loads. No error is shown in Scope — the pipeline just doesn't appear.

**Specific risk:** The current package is at `plugins/cellular_automata/`. When installed with `uv run daydream-scope install plugins/cellular_automata`, the package name in `pyproject.toml` determines the importable name. If `pyproject.toml` doesn't exist yet (it doesn't — only the `__main__.py` runner exists), `install` will fail.

**Required structure:**
```
plugins/cellular_automata/
├── pyproject.toml        ← MUST exist
├── cellular_automata/    ← package directory (or use src layout)
│   ├── __init__.py
│   ├── plugin.py         ← contains register_pipelines()
│   └── ...
```

**Prevention:**
1. **Match the entry point module path to the actual importable package name.** Test with:
   ```bash
   python -c "from cellular_automata.plugin import register_pipelines; print('OK')"
   ```
2. **Do not name the package `cellular_automata`** if there's any chance of collision with other packages. Use `daydream_ca` or `ca_scope_plugin`.
3. **Use the exact entry point group `"scope"`** — this is what Scope looks for. Not `"plugins"`, not `"scope.plugins"`.
4. **After install, verify with:** `uv run daydream-scope list-plugins` (if such command exists) or check Scope UI for the pipeline.

**Phase:** Plugin Wrapper (Phase 2).

---

### Pitfall 14: LFO Time Accumulation Without Real `dt` in Scope

**What goes wrong:** The LFO systems (Lenia mu/sigma/T breathing, GS feed oscillation) use `dt` (elapsed seconds) to advance phase. In the interactive viewer, `dt` comes from `pygame.time.Clock()`. In the Scope plugin, there is no clock — `__call__()` is called by Scope at irregular intervals.

**If you use a fixed `dt` per call** (e.g., `dt = 1/30.0` assuming 30fps), LFO breathing speed is wrong: at 15fps, LFO breathes twice as fast; at 60fps, half as fast.

**Why this matters:** The LFO period is tuned carefully (Lenia: ~70s, GS: ~30s). Wrong dt means LFO either runs too fast (frenetic) or too slow (no breathing).

**Prevention:**
1. **Use real elapsed time:** `dt = time.perf_counter() - self._last_call_time`
2. **Clamp dt to a safe range:** `dt = min(max(dt, 0.001), 0.1)` — prevents jumps if Scope pauses the pipeline.
3. **Pass dt to LFO system update:** `self.lfo_system.update(dt * lfo_speed)` — same pattern as existing viewer code.

**Phase:** Plugin Wrapper (Phase 2).

---

## Minor Pitfalls

---

### Pitfall 15: `np.take()` in `IridescentPipeline` Doesn't Port to CuPy Directly

**What goes wrong:** The 2D LUT lookup in `iridescent.py` uses:
```python
np.take(self._lut_2d, self._lut_indices.ravel(), axis=0, out=self.display_buffer.reshape(-1, 3))
```
CuPy has `cp.take()` but the `out=` parameter with pre-allocated buffer may behave differently. More importantly, `_lut_2d` is `(65536, 3)` uint8 — 192KB — which is tiny. This computation should stay on CPU.

**Prevention:**
1. **Keep the color pipeline (IridescentPipeline) on CPU.** The LUT lookup is memory-bandwidth bound, not compute-bound. 192KB fits in L2 cache. GPU benefits the FFT simulation, not the color LUT.
2. **The CuPy → CPU transfer for rendering:** After GPU sim step, transfer `world` array to CPU for color rendering:
   ```python
   world_cpu = cp.asnumpy(self.engine.world)  # GPU → CPU, ~1ms at 512x512
   rgb_uint8 = self.iridescent.render(world_cpu, ...)
   ```
   This transfer is ~0.3ms for 512×512 float32 — negligible.

**Phase:** CuPy Port (Phase 3).

---

### Pitfall 16: `__main__.py` Interactive Runner Breaks When Package Is Installed

**What goes wrong:** The current `python3 -m cellular_automata coral` runner uses `viewer.py` which uses pygame. Once the package is installed as a Scope plugin (via `pip install -e .` or `uv install`), running `python3 -m cellular_automata` may import the installed version instead of the local version, causing confusion. Worse, the `__main__.py` may be included in the installed package and conflict with Scope's discovery.

**Prevention:**
1. **Keep the interactive viewer and the Scope plugin as separate packages** or clearly separate entry points.
2. **Or: use a `src` layout.** `src/cellular_automata/` for the Scope-installable package; `plugins/cellular_automata/` for the interactive viewer. They share engine code via relative imports or a shared `ca_engines` subpackage.
3. **Exclude `viewer.py`, `controls.py`, `__main__.py` from the installed package** via `pyproject.toml`:
   ```toml
   [tool.hatch.build.targets.wheel]
   exclude = ["*/viewer.py", "*/controls.py", "*/__main__.py"]
   ```

**Phase:** Plugin Wrapper (Phase 2).

---

### Pitfall 17: RunPod Container May Not Have `scipy` Installed

**What goes wrong:** `viewer.py` has graceful fallback for missing scipy, but the CA engines assume numpy only. The `IridescentPipeline` does not depend on scipy. The viewer uses scipy for dilation/blur. In the Scope plugin, if `_build_color_offset()` is ported from the viewer, it uses `scipy.ndimage.zoom` — which may not be in the `daydreamlive/scope:main` container.

**Prevention:**
1. **Audit scipy usage in code that will run inside the Scope plugin** (not viewer.py). The engines (lenia.py, smoothlife.py, mnca.py, gray_scott.py, iridescent.py) do NOT use scipy — they are safe.
2. **The dilation and blur logic** (`_dilate_world`, `_blur_world`) lives in `viewer.py`. Do not port these to the Scope plugin unless needed.
3. **If scipy is needed for `_build_color_offset`:** Either include a scipy-free fallback (already exists in `viewer.py`), or add scipy to plugin dependencies.
4. **Check container at deploy time:**
   ```bash
   python -c "import scipy; print(scipy.__version__)"
   ```

**Phase:** RunPod Deployment (Phase 4).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Phase 2: Plugin Wrapper | Storing params in `__init__()` (#1) | Read ALL user controls from kwargs in `__call__()` |
| Phase 2: Plugin Wrapper | pygame import at module level (#2) | New `scope_pipeline.py` that never imports viewer.py or controls.py |
| Phase 2: Plugin Wrapper | THWC tensor shape/range (#7) | Unsqueeze(0), /255.0, .copy() — assert shape in dev |
| Phase 2: Plugin Wrapper | Warmup blocking init (#8) | Defer to first `__call__()` or background thread |
| Phase 2: Plugin Wrapper | Missing real `dt` for LFO (#14) | `time.perf_counter()` tracking between calls |
| Phase 2: Plugin Wrapper | Entry point misconfiguration (#13) | Write and test pyproject.toml before first install |
| Phase 3: CuPy Port | FFT precision differences (#3) | Visual QA on all 4 engines for 500+ steps |
| Phase 3: CuPy Port | `map_coordinates` wrap mode (#6) | Test wrap boundary or implement custom bilinear |
| Phase 3: CuPy Port | `scipy.ndimage.zoom` missing (#9) | Implement without zoom, build statics on CPU |
| Phase 3: CuPy Port | GS dead zone breaks (#12) | Explicit dead zone mask; verify after 1000 warmup steps |
| Phase 3: CuPy Port | VRAM budget design (#4) | 512x512, limit CuPy pool, profile with 14B model loaded |
| Phase 3: CuPy Port | Keep color pipeline on CPU (#15) | GPU sim → CPU transfer → CPU LUT → torch tensor |
| Phase 4: RunPod | CuPy CUDA version mismatch (#11) | Check nvcc version, install matching cupy-cuda12x |
| Phase 4: RunPod | VRAM OOM with 14B + CA (#4) | Profile VRAM with both loaded; use 512x512 on GPU |
| Phase 4: RunPod | scipy not in container (#17) | Audit, add to deps, or use fallbacks already in viewer.py |

---

## Confidence Assessment

| Area | Confidence | Source |
|------|------------|--------|
| Scope plugin contract (kwargs, THWC) | HIGH | Actual example_plugin code in this repo |
| pygame headless failure on RunPod | HIGH | Known SDL/X11 requirement, well-documented |
| CuPy rfft2 precision differences | MEDIUM | CuPy uses cuFFT (known), exact precision delta requires profiling |
| VRAM math (28GB for 14B float16) | HIGH | 14 billion × 2 bytes = 28GB; well-established |
| cupyx.scipy.ndimage.map_coordinates wrap mode | MEDIUM | Known to exist; mode support varies by version |
| GS dead zone sensitivity | HIGH | Directly derived from code: V*4.0 contrast means V=0.015 threshold is tight |
| Warmup blocking Scope init | MEDIUM | Pattern is documented; actual timeout behavior depends on Scope internals |

---

## Key Decisions This Research Informs

1. **Create a clean `scope_pipeline.py`** — no viewer.py, no controls.py, no pygame. Import only: engines, iridescent, presets, smoothing.
2. **Keep IridescentPipeline on CPU** — GPU sim step → CPU asnumpy → CPU LUT → torch tensor. Only FFT computation benefits from GPU.
3. **Use 512×512 for all engines on GPU** — same visual quality as 1024×1024 on CPU, 4× less VRAM, well within the 4GB headroom above the 14B model.
4. **Implement custom bilinear wrap** instead of depending on cupyx.scipy.ndimage.map_coordinates — eliminates version risk and is straightforward to implement.
5. **Build all static fields (containment, color offset, noise pool, flow fields) on CPU at init, transfer to GPU once.** Per-frame computation is FFT step + advection only.

---

*Research completed: 2026-02-17*
*Domain: Scope plugin wrapper + CuPy GPU acceleration + RunPod deployment*
*Focus: Integration pitfalls specific to adding these features to existing pygame/numpy CA simulation*
