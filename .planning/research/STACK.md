# Technology Stack: Scope Plugin + CuPy GPU + RunPod Deployment

**Project:** DayDream Scope - Cellular Automata Plugin (Scope Deployment Milestone)
**Researched:** 2026-02-17
**Confidence:** MEDIUM (Scope plugin API: HIGH from local code; CuPy/Blackwell compatibility: MEDIUM from training data + PyPI; headless pygame: HIGH from docs)

---

## Scope

This research covers only NEW stack requirements for three capabilities being added:

1. **Scope plugin wrapper** — packaging the CA as a text-only Scope pipeline
2. **CuPy GPU acceleration** — porting numpy/scipy to GPU for 30+ FPS at 1024x1024
3. **RunPod container deployment** — headless operation without pygame display

The existing stack (Python 3.10+, pygame-ce, numpy, scipy) is already validated. Do not re-evaluate it.

---

## Recommended Stack

### 1. Scope Plugin Packaging

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| hatchling | latest | Build backend | What the example plugin uses; `uv` resolves it automatically |
| torch | >=2.0.0 | THWC tensor output | Scope expects `torch.Tensor` not numpy arrays from `__call__()` |
| uv | system | Install plugin into Scope | `uv run daydream-scope install <path>` is the documented install path |

**Confidence: HIGH** — Verified directly from `/plugins/example_plugin/pyproject.toml` and `/plugins/example_plugin/pipeline.py`.

**Key integration constraint:** Scope pipelines must return `torch.Tensor` in THWC format with values in [0, 1]. The CA currently produces numpy arrays. The plugin wrapper needs exactly one conversion: `torch.from_numpy(rgb_array).unsqueeze(0)` to add the T dimension and return a (1, H, W, 3) tensor.

**pyproject.toml pattern (verified from example):**
```toml
[project]
name = "ca-scope-plugin"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    # Do NOT add pygame-ce — headless, no display
    # Do NOT add scipy in dependencies if using CuPy on GPU
]

[project.entry-points."scope"]
ca_plugin = "ca_plugin.plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["ca_plugin"]
```

**Plugin registration pattern (verified from example):**
```python
# plugin.py
from .pipeline import CAPipeline

def register_pipelines(registry):
    registry.register(
        name="cellular_automata",
        pipeline_class=CAPipeline,
        description="Bioluminescent cellular automata video source"
    )
```

**Text-only pipeline pattern (verified from example):**
```python
# pipeline.py — no prepare() method = text-only
class CAPipeline:
    def __init__(self, engine: str = "lenia", preset: str = "coral", ...):
        # Load-time: initialize CA engine, build FFT kernels
        # DO NOT import pygame here — headless
        ...

    def __call__(self, prompt: str, **kwargs):
        # Runtime params from kwargs ONLY
        preset = kwargs.get("preset", self._default_preset)
        speed = kwargs.get("speed", 1.0)
        hue_shift = kwargs.get("hue_shift", 0.0)

        # Step CA, render color
        frame_np = self._step_and_render()  # (H, W, 3) float32 [0,1]

        # Convert to THWC tensor (T=1 frame)
        tensor = torch.from_numpy(frame_np).unsqueeze(0)  # (1, H, W, 3)
        return tensor

    @staticmethod
    def ui_field_config():
        return {
            "preset": {"order": 1, "panel": "controls", "label": "Preset"},
            "speed": {"order": 2, "panel": "controls", "label": "Speed",
                      "min": 0.1, "max": 3.0, "step": 0.1},
            "hue_shift": {"order": 3, "panel": "controls", "label": "Hue Shift",
                          "min": 0.0, "max": 1.0, "step": 0.01},
        }
```

---

### 2. CuPy GPU Acceleration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| cupy-cuda12x | >=13.3.0 | GPU array library | Drop-in numpy replacement for CUDA 12.x; `cupyx.scipy.ndimage` has all needed ops |

**Confidence: MEDIUM** — CuPy 14.0.0 is the latest (confirmed from `pip index versions cupy`). `cupy-cuda12x` is the correct meta-package for CUDA 12. RTX 5090 (Blackwell, SM_100) support in CuPy 14 is based on training knowledge — verify before deployment.

**Install on RunPod (not locally):**
```bash
pip install cupy-cuda12x>=13.3.0
```

**Why cupy-cuda12x not cupy-cuda11x:** The RunPod container `daydreamlive/scope:main` bundles PyTorch which requires CUDA 12.x. Mixing CUDA versions causes conflicts. RTX 5090 requires CUDA 12.6+.

**What ports cleanly (near-identical API):**

| numpy/scipy call | CuPy equivalent | Notes |
|-----------------|-----------------|-------|
| `import numpy as np` | `import cupy as cp` | Module-level swap |
| `np.fft.rfft2(x)` | `cp.fft.rfft2(x)` | Identical signature |
| `np.fft.irfft2(x, s=...)` | `cp.fft.irfft2(x, s=...)` | Identical |
| `np.zeros()`, `np.ones()`, `np.exp()` | `cp.zeros()`, `cp.ones()`, `cp.exp()` | Identical |
| `np.clip()`, `np.roll()`, `np.pad()` | `cp.clip()`, `cp.roll()`, `cp.pad()` | Identical |
| `np.ogrid`, `np.meshgrid` | `cp.ogrid`, `cp.meshgrid` | Identical |
| `scipy.ndimage.gaussian_filter` | `cupyx.scipy.ndimage.gaussian_filter` | Identical signature |
| `scipy.ndimage.zoom` | `cupyx.scipy.ndimage.zoom` | Identical signature |
| `scipy.ndimage.maximum_filter` | `cupyx.scipy.ndimage.maximum_filter` | Identical signature |
| `scipy.ndimage.map_coordinates` | `cupyx.scipy.ndimage.map_coordinates` | Identical signature |

**Recommended porting strategy — numpy/cupy toggle:**
```python
# engine_base.py or a new compat.py
try:
    import cupy as xp
    import cupyx.scipy.ndimage as xpnd
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    import scipy.ndimage as xpnd
    GPU_AVAILABLE = False

# All engines use xp instead of np, xpnd instead of scipy.ndimage
# No other changes needed for 90% of the code
```

**Transfer points (only two needed):**
```python
# Input: seed patterns (CPU numpy → GPU cupy)
world_gpu = cp.asarray(seed_numpy)

# Output: final rendered frame (GPU cupy → CPU numpy → torch tensor)
frame_np = cp.asnumpy(frame_gpu)          # GPU → CPU
tensor = torch.from_numpy(frame_np).unsqueeze(0)  # numpy → THWC tensor
```

**Expected performance gain at 1024x1024 on RTX 5090:**
- Lenia FFT convolution: ~6-10x faster (GPU FFT is highly optimized)
- scipy.ndimage gaussian_filter: ~5-8x faster
- map_coordinates (flow fields): ~4-6x faster (memory-bound, less gain)
- Target: 30+ FPS is achievable for Lenia/SL/MNCA at 1024x1024

**Confidence: MEDIUM** — Based on typical GPU vs CPU speedup benchmarks for FFT and ndimage operations. Actual numbers depend on Blackwell driver optimization maturity.

---

### 3. Headless Operation (No pygame Display)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| (no new deps) | — | Drop pygame display | The Scope plugin renders to tensor; no pygame window or event loop needed |

**Confidence: HIGH** — The CA computation (engines, color pipeline) is pure numpy. Only `viewer.py` and `controls.py` use pygame. The plugin wrapper should import engines and `IridescentPipeline` directly, bypassing viewer entirely.

**What to NOT import in the plugin:**
```python
# DO NOT import in plugin context:
import pygame              # No display server on RunPod
from .viewer import CAViewer   # pygame-dependent
from .controls import ControlPanel  # pygame-dependent

# DO import:
from .lenia import Lenia
from .smoothlife import SmoothLife
from .mnca import MNCA
from .gray_scott import GrayScott
from .iridescent import IridescentPipeline
from .presets import get_preset, UNIFIED_ORDER
from .smoothing import SmoothedParameter
```

**Why this works:** `viewer.py` imports pygame at the top level, but all engine files (`lenia.py`, `smoothlife.py`, `mnca.py`, `gray_scott.py`, `iridescent.py`) are pure numpy — no pygame dependency. Import the engines directly.

---

### 4. RunPod Container

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| `daydreamlive/scope:main` | latest | Base image | Already running on the pod; has CUDA 12, PyTorch, Scope |
| cupy-cuda12x | >=13.3.0 | Added on top of base | pip install after pulling base image |

**No custom Dockerfile needed for initial deployment.** Install the plugin into the running Scope instance via SSH:

```bash
# On RunPod:
pip install cupy-cuda12x
cd /workspace
git clone https://github.com/diegochavez-io/cellular-automata_plug-in ca_plugin
uv run daydream-scope install /workspace/ca_plugin
```

**If a Dockerfile is needed later:**
```dockerfile
FROM daydreamlive/scope:main

# Install CuPy for CUDA 12 (RTX 5090 requires CUDA 12.6+)
RUN pip install cupy-cuda12x>=13.3.0

# Install CA plugin
COPY . /workspace/ca_plugin/
RUN uv run daydream-scope install /workspace/ca_plugin/
```

---

## Supporting Libraries (Do Not Add)

| Library | Why NOT |
|---------|---------|
| pygame-ce | Plugin is headless; adding it as a plugin dependency would break RunPod (no display) |
| scipy | On GPU use cupyx.scipy.ndimage; the toggle pattern handles CPU fallback |
| numba | More complex porting than CuPy, no drop-in numpy compatibility, CUDA compilation overhead |
| JAX | Different API from numpy (jit, grad transforms), requires rewrite not port |
| PyOpenGL/ModernGL | GPU compute via OpenGL is wrong layer; CuPy uses CUDA directly |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| cupy-cuda12x >=13.3.0 | CUDA 12.x, Python 3.10-3.12 | Meta-package; installs correct wheel automatically |
| cupy 14.0.0 (latest) | CUDA 12.x | Latest release as of PyPI check 2026-02-17 |
| torch >=2.0.0 | Python 3.10+ | Already in Scope base image |
| hatchling (any) | Python >=3.10 | Build-time only, resolved by uv |

**Critical check before deployment:** Verify RTX 5090 (Blackwell SM_100) is in CuPy 14.x supported GPU list. If not, CuPy will fall back to PTX compilation which is slower but still works. Run `python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"` on RunPod to verify.

---

## Alternatives Considered

| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| cupy-cuda12x | numba.cuda | Numba requires explicit CUDA kernel writing; CuPy is a drop-in numpy replacement |
| cupy-cuda12x | torch for compute | PyTorch tensors lack scipy.ndimage equivalents; FFT API differs; more rewrite needed |
| cupy-cuda12x | pycuda | Raw CUDA C; requires kernel authoring; far more work for same result |
| hatchling | setuptools | hatchling is what the example plugin uses; consistent toolchain |
| uv run daydream-scope install | pip install -e | Scope's install command handles entry point registration correctly |

---

## Installation

```bash
# On RunPod (inside running Scope container or SSH session):

# Step 1: Install CuPy
pip install "cupy-cuda12x>=13.3.0"

# Step 2: Verify GPU compute works
python3 -c "
import cupy as cp
import cupyx.scipy.ndimage as cpnd
a = cp.random.rand(1024, 1024, dtype=cp.float32)
b = cpnd.gaussian_filter(a, sigma=2.0)
print('CuPy OK, device:', cp.cuda.Device().id)
"

# Step 3: Install CA plugin
uv run daydream-scope install /workspace/ca_plugin/

# Local dev (CPU fallback, no CuPy needed):
# pip install -e plugins/cellular_automata/
# The xp/np toggle handles this transparently
```

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `cupy-cuda11x` | RTX 5090 requires CUDA 12.6+; CUDA 11 wheels won't load | `cupy-cuda12x` |
| `import pygame` in plugin | No display server in RunPod container; crashes on `pygame.init()` | Import engines directly |
| `scipy.ndimage` on GPU path | CPU-bound; will block GPU pipeline | `cupyx.scipy.ndimage` |
| `np.asarray(cupy_array)` | Correct but slow idiom | `cp.asnumpy(cupy_array)` — explicit and clear |
| `torch.tensor(cupy_array)` | Copies via CPU | `torch.as_tensor(cupy_array, device='cuda')` via DLPack (advanced) or just `torch.from_numpy(cp.asnumpy(frame))` for simplicity |
| Storing runtime params in `__init__` | Scope won't update them per-frame | Always read from `kwargs` in `__call__()` |

---

## Stack Patterns by Variant

**If GPU is available (RunPod with CuPy):**
- Use `xp = cupy`, `xpnd = cupyx.scipy.ndimage`
- All CA computation stays on GPU
- Single `cp.asnumpy()` call per frame at output

**If CPU only (local dev without CUDA):**
- Use `xp = numpy`, `xpnd = scipy.ndimage`
- Same code path, import toggle handles it
- pygame viewer still works for local visual testing

**If Scope plugin (headless, any hardware):**
- Never import pygame or viewer.py
- Return `torch.from_numpy(frame).unsqueeze(0)` — one T dimension
- Read all parameters from `kwargs` in `__call__()`

---

## Sources

- `/Users/agi/Code/daydream_scope/plugins/example_plugin/pyproject.toml` — entry point format, dependencies (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/pipeline.py` — THWC tensor output, ui_field_config structure, kwargs pattern (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/example_plugin/README.md` — install command `uv run daydream-scope install` (HIGH confidence)
- `/Users/agi/Code/daydream_scope/plugins/cellular_automata/viewer.py` — scipy function list; which files import pygame (HIGH confidence)
- `pip index versions cupy` — CuPy 14.0.0 is latest release as of 2026-02-17 (HIGH confidence)
- CuPy documentation (training knowledge) — cupyx.scipy.ndimage API compatibility, cupy-cuda12x naming (MEDIUM confidence; verify RTX 5090 SM_100 support before deploying)
- CLAUDE.md + MEMORY.md — RunPod container is `daydreamlive/scope:main`, RTX 5090 32GB VRAM (HIGH confidence)

---

*Stack research for: Scope plugin + CuPy GPU acceleration + RunPod deployment*
*Researched: 2026-02-17*
