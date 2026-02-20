"""
Cellular Automata Pipeline for DayDream Scope

Text-only pipeline that generates organic CA organisms as video frames.
No video input needed — the CA simulation is the video source.

The CA simulation runs in a background thread at ~15fps. When Krea
requests a frame via __call__, it grabs the latest one instantly.

Uses BasePipelineConfig + Pipeline ABC when Scope's formal API is available
(PLUG-02, PLUG-03). Falls back to plain class for local dev without Scope.
"""

import enum
import time
import threading
import torch
import numpy as np
from .simulator import CASimulator, FLOW_KEYS
from .presets import PRESETS


# ── Background CA Simulation Thread ──────────────────────────────────────

class _CABackgroundSim(threading.Thread):
    """Background thread that continuously steps the CA simulation.

    Keeps the latest frame available for Scope's pipeline __call__ to
    grab instantly. Without this, the CA would only advance when Krea
    requests a frame.

    Thread safety: sim_lock protects all simulator access. The background
    thread holds it during render_float(), and _ca_call holds it during
    set_runtime_params / apply_preset / run_warmup.
    """

    def __init__(self, simulator):
        super().__init__(daemon=True)
        self.simulator = simulator
        self._frame_lock = threading.Lock()
        self._latest_frame = None   # (H,W,3) float32 [0,1]
        self._running = True
        self._last_time = None
        self._target_fps = 15      # Background sim target
        self.sim_lock = threading.Lock()  # Protects simulator access

    def run(self):
        while self._running:
            now = time.perf_counter()
            if self._last_time is None:
                dt = 1.0 / self._target_fps
            else:
                dt = now - self._last_time
            dt = max(0.001, min(dt, 0.1))
            self._last_time = now

            try:
                with self.sim_lock:
                    frame = self.simulator.render_float(dt)
                with self._frame_lock:
                    self._latest_frame = frame
            except Exception:
                pass

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

def _ca_init(self, sim_size: int = 1024, preset: str = "coral", **kwargs):
    """Shared __init__ body for both formal and fallback CAPipeline.

    Creates CASimulator, runs warmup, and starts the background sim thread.

    Args:
        sim_size: Simulation grid resolution (512 or 1024).
        preset: Initial preset key (e.g. 'coral', 'medusa').
    """
    self.simulator = CASimulator(
        preset_key=preset, sim_size=sim_size, warmup=False,
        target_sps=30,  # Lower than viewer's 60 for CPU budget at 1024
    )
    # Run warmup immediately so first frames show developed structure
    self.simulator.run_warmup()

    # Start background simulation thread
    self._bg_sim = _CABackgroundSim(self.simulator)
    self._bg_sim.start()


def _ca_call(self, prompt: str = "", **kwargs) -> dict:
    """Shared __call__ body for both formal and fallback CAPipeline.

    Applies only CHANGED runtime params to the simulator. The background
    thread continuously steps and renders — _ca_call just tweaks knobs.

    Args:
        prompt: Ignored (text-only pipeline, no prompt needed).
        **kwargs: Runtime parameters from Scope UI.

    Returns:
        {"video": tensor} where tensor is (1, H, W, 3) float32 [0,1]
    """
    # --- Read ALL runtime params from kwargs every frame ---
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

    # --- Build dict of only CHANGED params (avoid spamming set_hue_offset every frame) ---
    if not hasattr(self, '_last_params'):
        self._last_params = {}

    changed = {}
    current = {
        "speed": speed, "hue": hue, "brightness": brightness,
        "thickness": thickness,
        "flow_radial": flow_radial, "flow_rotate": flow_rotate,
        "flow_swirl": flow_swirl, "flow_bubble": flow_bubble,
        "flow_ring": flow_ring, "flow_vortex": flow_vortex,
        "flow_vertical": flow_vertical,
    }
    for k, v in current.items():
        if self._last_params.get(k) != v:
            changed[k] = v
    self._last_params = current

    # Reseed is always applied when truthy (not tracked)
    if reseed:
        changed["reseed"] = True

    # --- Apply only changed params under lock (minimize lock hold time) ---
    if changed or (preset is not None and preset != self.simulator.preset_key):
        with self._bg_sim.sim_lock:
            if preset is not None and preset != self.simulator.preset_key:
                self.simulator.apply_preset(preset)
                self.simulator.run_warmup()

            if changed:
                self.simulator.set_runtime_params(**changed)

            # Tint: apply from kwargs only when user explicitly changed from default.
            # Otherwise let set_hue_offset (called via hue param) control tint.
            if tint_r != 1.0 or tint_g != 1.0 or tint_b != 1.0:
                self.simulator.iridescent.tint_r = tint_r
                self.simulator.iridescent.tint_g = tint_g
                self.simulator.iridescent.tint_b = tint_b

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
            default=1024,
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

else:
    # ── Fallback for local dev without Scope installed ────────────────

    class CAPipeline:
        """Fallback CA pipeline (plain class, no Scope API dependency)."""

        __init__ = _ca_init
        __call__ = _ca_call

        @staticmethod
        def ui_field_config():
            return {
                "sim_size": {
                    "order": 1, "panel": "settings", "label": "Sim Resolution",
                    "choices": [512, 1024], "default": 1024, "is_load_param": True,
                },
                "preset": {
                    "order": 1, "panel": "controls", "label": "Preset",
                    "choices": _PRESET_CHOICES,
                },
                "speed": {
                    "order": 2, "panel": "controls", "label": "Speed",
                    "min": 0.5, "max": 5.0, "step": 0.1,
                },
                "hue": {
                    "order": 3, "panel": "controls", "label": "Hue",
                    "min": 0.0, "max": 1.0, "step": 0.01,
                },
                "brightness": {
                    "order": 4, "panel": "controls", "label": "Brightness",
                    "min": 0.1, "max": 3.0, "step": 0.1,
                },
                "thickness": {
                    "order": 5, "panel": "controls", "label": "Thickness",
                    "min": 0.0, "max": 20.0, "step": 0.5,
                },
                "reseed": {
                    "order": 6, "panel": "controls", "label": "Reseed",
                    "type": "toggle",
                },
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
