"""
Plugin registration for DayDream Scope

Registers the Cellular Automata pipeline as a video source.
Uses @hookimpl decorator when Scope's formal plugin API is available,
falls back to the simpler registry pattern otherwise.
"""

try:
    from scope.core.plugins.hookspecs import hookimpl
except ImportError:
    hookimpl = None

from .pipeline import CAPipeline

# CAPreviewPipeline only exists when Scope API is available
try:
    from .pipeline import CAPreviewPipeline
except ImportError:
    CAPreviewPipeline = None


if hookimpl is not None:
    @hookimpl
    def register_pipelines(register):
        """Called when Scope loads the plugin (formal @hookimpl API)."""
        register(CAPipeline)
        if CAPreviewPipeline is not None:
            register(CAPreviewPipeline)
        # Install MJPEG preview middleware on Scope's port 8000 during startup
        # (before first request, so it's part of the initial middleware stack)
        from .pipeline import _install_preview_routes
        _install_preview_routes()
else:
    def register_pipelines(registry):
        """Called when Scope loads the plugin (simple registry API)."""
        registry.register(
            name="cellular_automata",
            pipeline_class=CAPipeline,
            description="Bioluminescent cellular automata organism as video source",
        )
