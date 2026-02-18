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


if hookimpl is not None:
    @hookimpl
    def register_pipelines(register):
        """Called when Scope loads the plugin (formal @hookimpl API)."""
        register(CAPipeline)
else:
    def register_pipelines(registry):
        """Called when Scope loads the plugin (simple registry API)."""
        registry.register(
            name="cellular_automata",
            pipeline_class=CAPipeline,
            description="Bioluminescent cellular automata organism as video source",
        )
