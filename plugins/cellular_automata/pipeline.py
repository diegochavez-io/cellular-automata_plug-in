"""
Cellular Automata Pipeline for DayDream Scope

Text-only pipeline that generates organic CA organisms as video frames.
No video input needed — the CA simulation is the video source.

Uses BasePipelineConfig + Pipeline ABC when Scope's formal API is available
(PLUG-02, PLUG-03). Falls back to plain class for local dev without Scope.
"""

import enum
import time
import threading
import io
import torch
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from .simulator import CASimulator, FLOW_KEYS
from .presets import PRESETS


# ── MJPEG Preview Server (raw CA output on port 8080) ────────────────────
# Opens in a second browser tab so you can see what the CA is generating
# before Krea transforms it. Runs as a daemon thread alongside Scope.

_mjpeg_server = None  # Module-level singleton

_PREVIEW_HTML = b'''<!DOCTYPE html>
<html><head><title>CA Preview</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#000; display:flex; justify-content:center;
       align-items:center; height:100vh; overflow:hidden; }
img { max-width:100vw; max-height:100vh; object-fit:contain; }
.label { position:fixed; top:12px; left:16px; color:#555;
         font:13px/1 monospace; pointer-events:none; }
</style></head>
<body>
<span class="label">CA Preview (raw input)</span>
<img src="/stream">
</body></html>'''


class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            while True:
                try:
                    with self.server.frame_lock:
                        jpeg = self.server.frame_jpeg
                    if jpeg:
                        header = (b'--frame\r\nContent-Type: image/jpeg\r\n'
                                  b'Content-Length: ' + str(len(jpeg)).encode()
                                  + b'\r\n\r\n')
                        self.wfile.write(header + jpeg + b'\r\n')
                    time.sleep(1.0 / 30)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(_PREVIEW_HTML)

    def log_message(self, *args):
        pass  # Silence HTTP request logs


def _start_mjpeg_server(port=8080):
    """Start the MJPEG preview server (singleton, idempotent)."""
    global _mjpeg_server
    if _mjpeg_server is not None:
        return
    try:
        server = HTTPServer(('0.0.0.0', port), _MJPEGHandler)
        server.frame_jpeg = None
        server.frame_lock = threading.Lock()
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        _mjpeg_server = server
        print(f"[CA] MJPEG preview started on port {port}")
    except OSError as e:
        print(f"[CA] MJPEG preview failed: {e}")


def _update_mjpeg_frame(frame_np):
    """Push a new frame to the MJPEG stream (called every render)."""
    if _mjpeg_server is None:
        return
    try:
        from PIL import Image
        img = Image.fromarray(
            (frame_np * 255).clip(0, 255).astype(np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        with _mjpeg_server.frame_lock:
            _mjpeg_server.frame_jpeg = buf.getvalue()
    except Exception:
        pass

# --- Preset choices: all headless-safe engine presets (internal keys) ---
_PRESET_CHOICES = [
    # Lenia presets
    "amoeba", "coral", "heartbeat", "jellyfish",
    "lava_lamp", "nebula", "tide_pool", "mycelium",
    # SmoothLife presets
    "sl_gliders", "sl_worms", "sl_elastic", "sl_pulse", "sl_chaos",
    # MNCA presets
    "mnca_soliton", "mnca_mitosis", "mnca_worm",
    "mnca_hunt", "mnca_coral",
    # Gray-Scott presets
    "reef", "deep_sea", "medusa", "labyrinth", "tentacles",
]


class PresetEnum(str, enum.Enum):
    """All headless-safe CA presets. Scope renders enum fields as dropdowns."""
    # Lenia
    amoeba = "amoeba"
    coral = "coral"
    heartbeat = "heartbeat"
    jellyfish = "jellyfish"
    lava_lamp = "lava_lamp"
    nebula = "nebula"
    tide_pool = "tide_pool"
    mycelium = "mycelium"
    # SmoothLife
    sl_gliders = "sl_gliders"
    sl_worms = "sl_worms"
    sl_elastic = "sl_elastic"
    sl_pulse = "sl_pulse"
    sl_chaos = "sl_chaos"
    # MNCA
    mnca_soliton = "mnca_soliton"
    mnca_mitosis = "mnca_mitosis"
    mnca_worm = "mnca_worm"
    mnca_hunt = "mnca_hunt"
    mnca_coral = "mnca_coral"
    # Gray-Scott
    reef = "reef"
    deep_sea = "deep_sea"
    medusa = "medusa"
    labyrinth = "labyrinth"
    tentacles = "tentacles"


# ── Shared __init__ and __call__ logic (used by both API branches) ────────

def _ca_init(self, sim_size: int = 512, preset: str = "coral", **kwargs):
    """Shared __init__ body for both formal and fallback CAPipeline.

    Creates CASimulator with warmup deferred to first __call__().
    Plugin loads instantly — no blocking warmup in __init__().

    Args:
        sim_size: Simulation grid resolution (512 or 1024).
        preset: Initial preset key (e.g. 'coral', 'medusa').
    """
    self.simulator = CASimulator(
        preset_key=preset, sim_size=sim_size, warmup=False
    )
    self._warmed_up = False
    self._last_time = None
    # Start MJPEG preview server (raw CA output on port 8080)
    _start_mjpeg_server()


def _ca_call(self, prompt: str = "", **kwargs) -> dict:
    """Shared __call__ body for both formal and fallback CAPipeline.

    All user-controllable params are read from kwargs every frame.
    Never stored from __init__().

    Args:
        prompt: Ignored (text-only pipeline, no prompt needed).
        **kwargs: Runtime parameters from Scope UI:
            preset (PresetEnum|str): CA preset key
            speed (float): Simulation speed multiplier
            hue (float): Hue offset 0-1
            brightness (float): Brightness multiplier
            thickness (float): Line dilation thickness
            reseed (bool): Trigger new organism seed
            flow_radial..flow_vertical (float): Flow field strengths -1 to 1
            tint_r, tint_g, tint_b (float): RGB tint multipliers 0-2

    Returns:
        {"video": tensor} where tensor is (1, H, W, 3) float32 [0,1]
    """
    # --- Deferred warmup (runs once on first call, not at plugin load) ---
    if not self._warmed_up:
        self.simulator.run_warmup()
        self._warmed_up = True

    # --- Read ALL runtime params from kwargs every frame (PLUG-04) ---
    # Pydantic defaults match coral preset, so untouched sliders = coral values.
    preset = kwargs.get("preset", None)
    speed = kwargs.get("speed", 1.0)
    hue = kwargs.get("hue", 0.25)
    brightness = kwargs.get("brightness", 1.0)
    thickness = kwargs.get("thickness", 0.0)
    reseed = kwargs.get("reseed", False)

    # Flow field strengths (defaults match coral preset via Pydantic)
    flow_radial = kwargs.get("flow_radial", -0.10)
    flow_rotate = kwargs.get("flow_rotate", 0.40)
    flow_swirl = kwargs.get("flow_swirl", 0.35)
    flow_bubble = kwargs.get("flow_bubble", 0.15)
    flow_ring = kwargs.get("flow_ring", 0.0)
    flow_vortex = kwargs.get("flow_vortex", 0.25)
    flow_vertical = kwargs.get("flow_vertical", 0.0)

    # Tint RGB multipliers (applied after hue sets base tint)
    tint_r = kwargs.get("tint_r", 1.0)
    tint_g = kwargs.get("tint_g", 1.0)
    tint_b = kwargs.get("tint_b", 1.0)

    # --- Preset change detection ---
    # Handle both PresetEnum and raw string values
    if preset is not None:
        preset = getattr(preset, 'value', preset)
    if preset is not None and preset != self.simulator.preset_key:
        self.simulator.apply_preset(preset)
        # Re-warmup for new engine (apply_preset skips warmup when
        # self._warmup is False, but we want warmup after preset switch)
        self.simulator.run_warmup()

    # --- Apply runtime params to simulator ---
    self.simulator.set_runtime_params(
        speed=speed,
        hue=hue,
        brightness=brightness,
        thickness=thickness,
        reseed=reseed,
        flow_radial=flow_radial,
        flow_rotate=flow_rotate,
        flow_swirl=flow_swirl,
        flow_bubble=flow_bubble,
        flow_ring=flow_ring,
        flow_vortex=flow_vortex,
        flow_vertical=flow_vertical,
    )

    # --- Tint: multiply on top of hue-derived tint (1.0 = no change) ---
    self.simulator.iridescent.tint_r *= tint_r
    self.simulator.iridescent.tint_g *= tint_g
    self.simulator.iridescent.tint_b *= tint_b

    # --- Wall-clock dt for LFO accuracy (PLUG-06) ---
    now = time.perf_counter()
    if self._last_time is None:
        dt = 1.0 / 30.0  # First frame: assume 30fps
    else:
        dt = now - self._last_time
    dt = max(0.001, min(dt, 0.1))  # Clamp to [0.001, 0.1]
    self._last_time = now

    # --- Render one frame (PLUG-05) ---
    frame_np = self.simulator.render_float(dt)  # (H, W, 3) float32 [0,1]

    # Push raw CA frame to MJPEG preview (before Krea transforms it)
    _update_mjpeg_frame(frame_np)

    # Convert to THWC torch tensor: (1, H, W, 3) float32
    # .copy() is required: render_float may share internal buffer memory
    tensor = torch.from_numpy(frame_np.copy()).unsqueeze(0)

    return {"video": tensor}


# ── Try formal Scope API (BasePipelineConfig + Pipeline ABC) ──────────────

try:
    from pydantic import Field
    from scope.core.pipelines.base_schema import (
        BasePipelineConfig, ModeDefaults, UsageType, ui_field_config,
    )
    from scope.core.pipelines.interface import Pipeline
    _HAS_SCOPE_API = True
except ImportError:
    _HAS_SCOPE_API = False


if _HAS_SCOPE_API:
    # ── Formal Scope API branch (PLUG-02, PLUG-03) ───────────────────

    class CAPipelineConfig(BasePipelineConfig):
        pipeline_id = "cellular-automata"
        pipeline_name = "Cellular Automata"
        pipeline_description = (
            "Bioluminescent cellular automata organism as video source"
        )
        supports_prompts = False
        usage = [UsageType.PREPROCESSOR]
        modes = {"video": ModeDefaults(default=True)}

        # Load-time
        sim_size: int = Field(
            default=512,
            description="Simulation grid resolution",
            json_schema_extra=ui_field_config(
                order=1, label="Sim Resolution", is_load_param=True,
            ),
        )
        # Runtime
        preset: PresetEnum = Field(
            default=PresetEnum.coral,
            description="CA preset to run",
            json_schema_extra=ui_field_config(order=1, label="Preset"),
        )
        speed: float = Field(
            default=1.0, ge=0.5, le=5.0,
            json_schema_extra=ui_field_config(order=2, label="Speed"),
        )
        hue: float = Field(
            default=0.25, ge=0.0, le=1.0,
            json_schema_extra=ui_field_config(order=3, label="Hue"),
        )
        brightness: float = Field(
            default=1.0, ge=0.1, le=3.0,
            json_schema_extra=ui_field_config(order=4, label="Brightness"),
        )
        thickness: float = Field(
            default=0.0, ge=0.0, le=20.0,
            json_schema_extra=ui_field_config(order=5, label="Thickness"),
        )
        reseed: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(order=6, label="Reseed"),
        )
        # Flow field strengths (defaults match coral preset)
        flow_radial: float = Field(
            default=-0.10, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=10, label="Flow Radial"),
        )
        flow_rotate: float = Field(
            default=0.40, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=11, label="Flow Rotate"),
        )
        flow_swirl: float = Field(
            default=0.35, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=12, label="Flow Swirl"),
        )
        flow_bubble: float = Field(
            default=0.15, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=13, label="Flow Bubble"),
        )
        flow_ring: float = Field(
            default=0.0, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=14, label="Flow Ring"),
        )
        flow_vortex: float = Field(
            default=0.25, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=15, label="Flow Vortex"),
        )
        flow_vertical: float = Field(
            default=0.0, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=16, label="Flow Vertical"),
        )
        # Tint RGB multipliers
        tint_r: float = Field(
            default=1.0, ge=0.0, le=2.0,
            json_schema_extra=ui_field_config(order=20, label="Tint R"),
        )
        tint_g: float = Field(
            default=1.0, ge=0.0, le=2.0,
            json_schema_extra=ui_field_config(order=21, label="Tint G"),
        )
        tint_b: float = Field(
            default=1.0, ge=0.0, le=2.0,
            json_schema_extra=ui_field_config(order=22, label="Tint B"),
        )

    from scope.core.pipelines.interface import Requirements as _Requirements

    class CAPipeline(Pipeline):
        """Formal Scope pipeline using BasePipelineConfig + Pipeline ABC."""

        @classmethod
        def get_config_class(cls):
            return CAPipelineConfig

        def prepare(self, **kwargs):
            """Declare video input so Scope treats CA as a valid preprocessor.

            The actual video input is ignored — CA generates its own frames.
            This allows CA to appear in the Preprocessor dropdown for pipelines
            like krea-realtime-video, enabling the chain: CA → Krea RT + LoRAs.
            """
            return _Requirements(input_size=1)

        __init__ = _ca_init
        __call__ = _ca_call

    # ── Standalone preview pipeline (appears in Pipeline ID dropdown) ──

    class CAPreviewConfig(BasePipelineConfig):
        pipeline_id = "cellular-automata-preview"
        pipeline_name = "CA Preview"
        pipeline_description = "Raw cellular automata output (no AI processing)"
        supports_prompts = False
        # Video mode so Scope UI shows controls properly
        modes = {"video": ModeDefaults(default=True)}

        sim_size: int = Field(
            default=512,
            description="Simulation grid resolution",
            json_schema_extra=ui_field_config(
                order=1, label="Sim Resolution", is_load_param=True,
            ),
        )
        preset: PresetEnum = Field(
            default=PresetEnum.coral,
            description="CA preset to run",
            json_schema_extra=ui_field_config(order=1, label="Preset"),
        )
        speed: float = Field(
            default=1.0, ge=0.5, le=5.0,
            json_schema_extra=ui_field_config(order=2, label="Speed"),
        )
        hue: float = Field(
            default=0.25, ge=0.0, le=1.0,
            json_schema_extra=ui_field_config(order=3, label="Hue"),
        )
        brightness: float = Field(
            default=1.0, ge=0.1, le=3.0,
            json_schema_extra=ui_field_config(order=4, label="Brightness"),
        )
        thickness: float = Field(
            default=0.0, ge=0.0, le=20.0,
            json_schema_extra=ui_field_config(order=5, label="Thickness"),
        )
        reseed: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(order=6, label="Reseed"),
        )
        # Flow field strengths (defaults match coral preset)
        flow_radial: float = Field(
            default=-0.10, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=10, label="Flow Radial"),
        )
        flow_rotate: float = Field(
            default=0.40, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=11, label="Flow Rotate"),
        )
        flow_swirl: float = Field(
            default=0.35, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=12, label="Flow Swirl"),
        )
        flow_bubble: float = Field(
            default=0.15, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=13, label="Flow Bubble"),
        )
        flow_ring: float = Field(
            default=0.0, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=14, label="Flow Ring"),
        )
        flow_vortex: float = Field(
            default=0.25, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=15, label="Flow Vortex"),
        )
        flow_vertical: float = Field(
            default=0.0, ge=-1.0, le=1.0,
            json_schema_extra=ui_field_config(order=16, label="Flow Vertical"),
        )
        # Tint RGB multipliers
        tint_r: float = Field(
            default=1.0, ge=0.0, le=2.0,
            json_schema_extra=ui_field_config(order=20, label="Tint R"),
        )
        tint_g: float = Field(
            default=1.0, ge=0.0, le=2.0,
            json_schema_extra=ui_field_config(order=21, label="Tint G"),
        )
        tint_b: float = Field(
            default=1.0, ge=0.0, le=2.0,
            json_schema_extra=ui_field_config(order=22, label="Tint B"),
        )

    class CAPreviewPipeline(Pipeline):
        """Standalone pipeline for previewing raw CA output."""

        @classmethod
        def get_config_class(cls):
            return CAPreviewConfig

        def prepare(self, **kwargs):
            """Declare video input requirement so Scope UI shows properly.
            The actual video input is ignored — CA generates its own frames.
            """
            return _Requirements(input_size=1)

        __init__ = _ca_init
        __call__ = _ca_call

else:
    # ── Fallback for local dev without Scope installed ────────────────

    class CAPipeline:
        """Fallback CA pipeline (plain class, no Scope API dependency)."""

        __init__ = _ca_init
        __call__ = _ca_call

        @staticmethod
        def ui_field_config():
            """Configure how parameters appear in Scope UI (PLUG-02 fallback).

            Returns:
                dict: UI configuration for each parameter.
                      Load-time params go in 'settings' panel.
                      Runtime params go in 'controls' panel.
            """
            return {
                # --- Load-time (Settings panel — requires pipeline reload) ---
                "sim_size": {
                    "order": 1,
                    "panel": "settings",
                    "label": "Sim Resolution",
                    "choices": [512, 1024],
                    "is_load_param": True,
                },
                # --- Runtime (Controls panel — updates live per-frame) ---
                "preset": {
                    "order": 1,
                    "panel": "controls",
                    "label": "Preset",
                    "choices": _PRESET_CHOICES,
                },
                "speed": {
                    "order": 2,
                    "panel": "controls",
                    "label": "Speed",
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                },
                "hue": {
                    "order": 3,
                    "panel": "controls",
                    "label": "Hue",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                },
                "brightness": {
                    "order": 4,
                    "panel": "controls",
                    "label": "Brightness",
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                },
                "thickness": {
                    "order": 5,
                    "panel": "controls",
                    "label": "Thickness",
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                },
                "reseed": {
                    "order": 6,
                    "panel": "controls",
                    "label": "Reseed",
                    "type": "toggle",
                },
                # Flow field strengths
                "flow_radial": {
                    "order": 10, "panel": "controls", "label": "Flow Radial",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                "flow_rotate": {
                    "order": 11, "panel": "controls", "label": "Flow Rotate",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                "flow_swirl": {
                    "order": 12, "panel": "controls", "label": "Flow Swirl",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                "flow_bubble": {
                    "order": 13, "panel": "controls", "label": "Flow Bubble",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                "flow_ring": {
                    "order": 14, "panel": "controls", "label": "Flow Ring",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                "flow_vortex": {
                    "order": 15, "panel": "controls", "label": "Flow Vortex",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                "flow_vertical": {
                    "order": 16, "panel": "controls", "label": "Flow Vertical",
                    "min": -1.0, "max": 1.0, "step": 0.05,
                },
                # Tint RGB multipliers
                "tint_r": {
                    "order": 20, "panel": "controls", "label": "Tint R",
                    "min": 0.0, "max": 2.0, "step": 0.05,
                },
                "tint_g": {
                    "order": 21, "panel": "controls", "label": "Tint G",
                    "min": 0.0, "max": 2.0, "step": 0.05,
                },
                "tint_b": {
                    "order": 22, "panel": "controls", "label": "Tint B",
                    "min": 0.0, "max": 2.0, "step": 0.05,
                },
            }
