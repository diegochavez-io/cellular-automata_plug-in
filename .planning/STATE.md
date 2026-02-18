# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Beautiful, constantly-evolving organic imagery that feeds Scope's pipeline — every system alive, iridescent, never black.
**Current focus:** Milestone v2.0 — Scope Deployment

## Current Position

Phase: 3 of 6 - Extract CASimulator
Plan: 2 of 2 (03-01 complete, 03-02 complete)
Status: Phase 3 complete — CASimulator extracted, viewer delegating, snap() simplified
Last activity: 2026-02-18 — Plan 03-02 executed (viewer.py refactored to thin pygame wrapper)

## Progress

[████░░░░░░] 40% — Phase 3 complete (2 of 2 plans done)

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

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-02-18
Stopped at: Completed 03-02-PLAN.md (viewer.py refactored). Phase 3 complete. Next: Phase 4 — Scope plugin wrapper.
Resume file: None
