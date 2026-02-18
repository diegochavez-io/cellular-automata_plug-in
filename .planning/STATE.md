# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Beautiful, constantly-evolving organic imagery that feeds Scope's pipeline — every system alive, iridescent, never black.
**Current focus:** Milestone v2.0 — Scope Deployment

## Current Position

Phase: 4 of 6 - Scope Plugin Wrapper
Plan: 1 of 2 (04-01 complete)
Status: Phase 4, Plan 1 complete — plugin package created (pyproject.toml, plugin.py, pipeline.py, warmup support)
Last activity: 2026-02-18 — Plan 04-01 executed (Scope plugin wrapper: pyproject.toml, plugin.py, pipeline.py)

## Progress

[█████░░░░░] 50% — Phase 4 Plan 1 complete (1 of 2 plans in phase 4)

## Accumulated Context

### Decisions

- v2.0 = full pipeline: Scope plugin + CuPy GPU + RunPod deployment
- Phase numbering continues from v1.0 (phases 1-2 complete, 3-6 are v2.0)
- Old v1.0 phases 3-6 (parameter safety, UI cleanup, new engines) deprioritized
- 512x512 on GPU (not 1024) — saves VRAM, GPU speed compensates
- Keep IridescentPipeline on CPU — LUT is cache-bound
- Custom bilinear wrap instead of cupyx.scipy.ndimage.map_coordinates
- Scope plugin uses BasePipelineConfig with Pydantic fields + ui_field_config
- Registration uses @hookimpl + register(PipelineClass)
- __call__() returns {"video": tensor} dict, not raw tensor
- Enum/dropdown IS supported in Scope UI for preset selector
- Context7 MCP connected and verified (DayDream Scope + RunPod docs)
- CASimulator does NOT auto-resize sim_size — caller controls size (unlike viewer.py)
- ENGINE_CLASSES in simulator.py restricted to headless-safe engines (lenia, gray_scott, smoothlife, mnca)
- _render_frame() in CASimulator returns raw (H,W,3) uint8 numpy array, not pygame.Surface
- viewer.py is a thin pygame wrapper: self.simulator = CASimulator(preset, sim_size); run loop calls self.simulator.step(dt)
- ENGINE_LABELS kept in viewer.py (display-only, not simulation logic)
- FLOW_SLIDER_DEFS kept in viewer.py (panel layout config, not sim logic)
- Dual-pattern Scope API: try BasePipelineConfig+Pipeline ABC, fall back to plain class — single codebase works in Scope and bare Python
- Deferred warmup: CASimulator(warmup=False) in __init__, run_warmup() fires on first __call__() — plugin loads instantly
- Shared _ca_init/_ca_call module-level helpers used as class method bodies in both branches
- _PRESET_CHOICES lists all 23 headless-safe presets by internal key for Scope UI

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 04-01-PLAN.md (Scope plugin wrapper files created). Next: 04-02-PLAN.md.
Resume file: None
