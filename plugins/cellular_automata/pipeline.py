"""
Cellular Automata Pipeline for DayDream Scope

Text-only pipeline that generates organic CA organisms as video frames.
No video input needed — the CA simulation is the video source.

Uses BasePipelineConfig + Pipeline ABC when Scope's formal API is available
(PLUG-02, PLUG-03). Falls back to plain class for local dev without Scope.
"""

import time
import torch
import numpy as np
from .simulator import CASimulator

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


def _ca_call(self, prompt: str = "", **kwargs) -> dict:
    """Shared __call__ body for both formal and fallback CAPipeline.

    All user-controllable params are read from kwargs every frame.
    Never stored from __init__().

    Args:
        prompt: Ignored (text-only pipeline, no prompt needed).
        **kwargs: Runtime parameters from Scope UI:
            preset (str): CA preset key
            speed (float): Simulation speed multiplier
            hue (float): Hue offset 0-1
            brightness (float): Brightness multiplier
            thickness (float): Line dilation thickness
            reseed (bool): Trigger new organism seed

    Returns:
        {"video": tensor} where tensor is (1, H, W, 3) float32 [0,1]
    """
    # --- Deferred warmup (runs once on first call, not at plugin load) ---
    if not self._warmed_up:
        self.simulator.run_warmup()
        self._warmed_up = True

    # --- Read ALL runtime params from kwargs every frame (PLUG-04) ---
    preset = kwargs.get("preset", None)
    speed = kwargs.get("speed", 1.0)
    hue = kwargs.get("hue", 0.25)
    brightness = kwargs.get("brightness", 1.0)
    thickness = kwargs.get("thickness", 0.0)
    reseed = kwargs.get("reseed", False)

    # --- Preset change detection ---
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
    )

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
        preset: str = Field(
            default="coral",
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
        # No usage set → appears as standalone pipeline in Pipeline ID dropdown
        modes = {"text": ModeDefaults(default=True)}

        sim_size: int = Field(
            default=512,
            description="Simulation grid resolution",
            json_schema_extra=ui_field_config(
                order=1, label="Sim Resolution", is_load_param=True,
            ),
        )
        preset: str = Field(
            default="coral",
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

    class CAPreviewPipeline(Pipeline):
        """Standalone pipeline for previewing raw CA output."""

        @classmethod
        def get_config_class(cls):
            return CAPreviewConfig

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
            }
