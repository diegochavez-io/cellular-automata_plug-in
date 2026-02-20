"""
Cellular Automata Pipeline for DayDream Scope

Text-only pipeline that generates organic CA organisms as video frames.
No video input needed — the CA simulation is the video source.

The CA simulation runs in a background thread, continuously generating
frames for the MJPEG preview. When Krea requests a frame, it grabs
the latest one from the background sim — zero lag.

Preview is served through Scope's own port 8000 at /api/v1/ca-preview
so it works through RunPod proxy with zero extra port configuration.
Fallback: standalone MJPEG server on port 8080 for local dev.

Uses BasePipelineConfig + Pipeline ABC when Scope's formal API is available
(PLUG-02, PLUG-03). Falls back to plain class for local dev without Scope.
"""

import asyncio
import enum
import time
import threading
import io
import torch
import numpy as np
from .simulator import CASimulator, FLOW_KEYS
from .presets import PRESETS


# ── MJPEG Preview Frame Store ────────────────────────────────────────────
# Shared frame buffer used by both the Scope route and the fallback server.

_frame_jpeg = None      # Latest JPEG bytes
_frame_lock = threading.Lock()

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
<img src="/api/v1/ca-preview/stream">
</body></html>'''


def _update_mjpeg_frame(frame_np):
    """Push a new frame to the preview stream (called every render)."""
    global _frame_jpeg
    try:
        from PIL import Image
        img = Image.fromarray(
            (frame_np * 255).clip(0, 255).astype(np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        with _frame_lock:
            _frame_jpeg = buf.getvalue()
    except Exception:
        pass


# ── Scope ASGI Interceptor (serves preview on port 8000) ─────────────────
# Patches Scope's middleware stack build to inject our preview handler at
# the outermost ASGI layer — before any router or SPA catch-all can
# swallow the request. Works on the same port and proxy URL, zero config.

_preview_middleware_installed = False


async def _serve_preview_html(send):
    """Serve the CA preview HTML page."""
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            [b"content-type", b"text/html"],
            [b"cache-control", b"no-cache"],
        ],
    })
    await send({
        "type": "http.response.body",
        "body": _PREVIEW_HTML,
    })


async def _serve_preview_stream(receive, send):
    """Serve the MJPEG stream of raw CA frames."""
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [
            [b"content-type",
             b"multipart/x-mixed-replace; boundary=frame"],
            [b"cache-control", b"no-cache"],
        ],
    })
    disconnected = False

    async def _watch_disconnect():
        nonlocal disconnected
        while not disconnected:
            msg = await receive()
            if msg.get("type") == "http.disconnect":
                disconnected = True
                return

    asyncio.ensure_future(_watch_disconnect())
    try:
        while not disconnected:
            with _frame_lock:
                jpeg = _frame_jpeg
            if jpeg:
                chunk = (b'--frame\r\nContent-Type: image/jpeg\r\n'
                         b'Content-Length: '
                         + str(len(jpeg)).encode()
                         + b'\r\n\r\n' + jpeg + b'\r\n')
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                })
            await asyncio.sleep(1.0 / 30)
    except Exception:
        pass


def _install_preview_routes():
    """Patch Scope's ASGI stack to serve CA preview (idempotent).

    Uses a deferred approach: a background thread waits for the server to
    fully start, then directly replaces the middleware_stack callable with
    a wrapper that intercepts /api/v1/ca-preview requests.
    """
    global _preview_middleware_installed
    if _preview_middleware_installed:
        return
    _preview_middleware_installed = True  # Prevent duplicate installs

    _log_path = "/workspace/ca_preview_debug.log"

    def _log(msg):
        with open(_log_path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            f.flush()

    _log("_install_preview_routes() called")

    def _deferred_patch():
        """Wait for server to be ready, then patch the ASGI stack."""
        _log("Deferred thread started, sleeping 8s...")
        time.sleep(8)
        try:
            _log("Importing scope.server.app...")
            from scope.server.app import app as scope_app
            _log(f"Got app: type={type(scope_app).__name__}, "
                 f"class_mro={[c.__name__ for c in type(scope_app).__mro__]}, "
                 f"middleware_stack_is_none={scope_app.middleware_stack is None}")

            # Force a middleware stack build if not done yet
            if scope_app.middleware_stack is None:
                _log("Building middleware stack...")
                scope_app.middleware_stack = scope_app.build_middleware_stack()
                _log(f"Stack built: {type(scope_app.middleware_stack)}")

            _orig_stack = scope_app.middleware_stack
            _log(f"Original stack type: {type(_orig_stack)}")

            async def _ca_patched_stack(asgi_scope, receive, send):
                if asgi_scope["type"] == "http":
                    path = asgi_scope.get("path", "")
                    if path == "/api/v1/ca-preview":
                        await _serve_preview_html(send)
                        return
                    if path == "/api/v1/ca-preview/stream":
                        await _serve_preview_stream(receive, send)
                        return
                await _orig_stack(asgi_scope, receive, send)

            scope_app.middleware_stack = _ca_patched_stack
            _log(f"PATCHED! middleware_stack is now: {type(scope_app.middleware_stack)}")

            # Verify the patch sticks
            time.sleep(1)
            _log(f"Verify: middleware_stack type={type(scope_app.middleware_stack).__name__}, "
                 f"is_our_func={scope_app.middleware_stack is _ca_patched_stack}")
        except Exception as e:
            import traceback
            _log(f"FAILED: {e}\n{traceback.format_exc()}")
            _start_fallback_mjpeg_server()

    threading.Thread(target=_deferred_patch, daemon=True).start()
    _log("Thread launched")


# ── Fallback MJPEG Server (port 8080, local dev only) ────────────────────

_fallback_server = None


def _start_fallback_mjpeg_server(port=8080):
    """Start standalone MJPEG server when Scope routes aren't available."""
    global _fallback_server
    if _fallback_server is not None:
        return
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/stream':
                    self.send_response(200)
                    self.send_header('Content-Type',
                                     'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    while True:
                        try:
                            with _frame_lock:
                                jpeg = _frame_jpeg
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
                    # For fallback, point stream URL to local port
                    html = _PREVIEW_HTML.replace(
                        b'/api/v1/ca-preview/stream', b'/stream')
                    self.wfile.write(html)

            def log_message(self, *args):
                pass

        server = HTTPServer(('0.0.0.0', port), _Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        _fallback_server = server
        print(f"[CA] Fallback MJPEG preview on port {port}")
    except OSError as e:
        print(f"[CA] Fallback MJPEG server failed: {e}")


# ── Background CA Simulation Thread ──────────────────────────────────────
# Runs the CA simulation continuously so the MJPEG preview always has
# fresh frames, even when no WebRTC session is active.

class _CABackgroundSim(threading.Thread):
    """Background thread that continuously steps the CA simulation.

    Pushes every frame to the MJPEG server and keeps the latest frame
    available for Scope's pipeline __call__ to grab instantly.
    """

    def __init__(self, simulator):
        super().__init__(daemon=True)
        self.simulator = simulator
        self._frame_lock = threading.Lock()
        self._latest_frame = None   # (H,W,3) float32 [0,1]
        self._running = True
        self._last_time = None
        self._target_fps = 20      # Background sim target framerate

    def run(self):
        print("[CA] Background simulation thread started")
        while self._running:
            now = time.perf_counter()
            if self._last_time is None:
                dt = 1.0 / self._target_fps
            else:
                dt = now - self._last_time
            dt = max(0.001, min(dt, 0.1))
            self._last_time = now

            try:
                frame = self.simulator.render_float(dt)
                with self._frame_lock:
                    self._latest_frame = frame
                _update_mjpeg_frame(frame)
            except Exception as e:
                print(f"[CA] Background sim error: {e}")

            elapsed = time.perf_counter() - now
            sleep_time = max(0, (1.0 / self._target_fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self):
        """Return the most recent frame (H,W,3) float32 [0,1] or None."""
        with self._frame_lock:
            return self._latest_frame

    def stop(self):
        self._running = False

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

    Creates CASimulator, runs warmup, and starts the background sim thread.
    The background thread continuously generates frames for the MJPEG preview
    so you can see the raw CA output even without an active WebRTC session.

    Args:
        sim_size: Simulation grid resolution (512 or 1024).
        preset: Initial preset key (e.g. 'coral', 'medusa').
    """
    self.simulator = CASimulator(
        preset_key=preset, sim_size=sim_size, warmup=False
    )
    # Run warmup immediately so first frames show developed structure
    self.simulator.run_warmup()

    # Install preview routes on Scope's port 8000 (or fallback to port 8080)
    _install_preview_routes()

    # Start background simulation thread — preview starts immediately
    self._bg_sim = _CABackgroundSim(self.simulator)
    self._bg_sim.start()


def _ca_call(self, prompt: str = "", **kwargs) -> dict:
    """Shared __call__ body for both formal and fallback CAPipeline.

    Updates runtime params on the simulator (background thread picks them
    up on next frame), then grabs the latest frame from the background sim.

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
    # --- Read ALL runtime params from kwargs every frame (PLUG-04) ---
    preset = kwargs.get("preset", None)
    speed = kwargs.get("speed", 1.0)
    hue = kwargs.get("hue", 0.25)
    brightness = kwargs.get("brightness", 1.0)
    thickness = kwargs.get("thickness", 0.0)
    reseed = kwargs.get("reseed", False)

    flow_radial = kwargs.get("flow_radial", -0.10)
    flow_rotate = kwargs.get("flow_rotate", 0.40)
    flow_swirl = kwargs.get("flow_swirl", 0.35)
    flow_bubble = kwargs.get("flow_bubble", 0.15)
    flow_ring = kwargs.get("flow_ring", 0.0)
    flow_vortex = kwargs.get("flow_vortex", 0.25)
    flow_vertical = kwargs.get("flow_vertical", 0.0)

    tint_r = kwargs.get("tint_r", 1.0)
    tint_g = kwargs.get("tint_g", 1.0)
    tint_b = kwargs.get("tint_b", 1.0)

    # --- Preset change detection ---
    if preset is not None:
        preset = getattr(preset, 'value', preset)
    if preset is not None and preset != self.simulator.preset_key:
        self.simulator.apply_preset(preset)
        self.simulator.run_warmup()

    # --- Apply runtime params (background thread reads these on next frame) ---
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

    self.simulator.iridescent.tint_r *= tint_r
    self.simulator.iridescent.tint_g *= tint_g
    self.simulator.iridescent.tint_b *= tint_b

    # --- Grab latest frame from background sim (no sim work here) ---
    frame_np = self._bg_sim.get_latest_frame()
    if frame_np is None:
        # Background sim hasn't produced a frame yet — return black
        h = w = self.simulator.sim_size
        frame_np = np.zeros((h, w, 3), dtype=np.float32)

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
