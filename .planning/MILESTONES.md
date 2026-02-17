# Milestones

## v1.0 — Local Prototype (Complete)

**Shipped:** 2026-02-17
**Commit:** `0d009e3`
**Phases:** 1-2 completed (3-6 deprioritized — safe params, preset cleanup, new engines deferred to post-deployment)

**What shipped:**
- 4 CA engines: Lenia, SmoothLife, MNCA, Gray-Scott
- 11 curated presets in unified order
- Multi-zone iridescent color pipeline (2D LUT, cosine palettes, 2nd+3rd harmonics)
- Universal flow fields (7 types, 0.8px/step semi-Lagrangian advection)
- Varied blob seeds, radial containment, auto-reseed
- Bloom, edge-weighted color mapping, alpha depth curve
- Performance optimized (float32, pre-computed noise pool)
- Interactive pygame-ce viewer with controls

**Deferred from v1.0:**
- Safe sliders (EMA smoothing) — started, not shipped
- Simplified control panel
- New engines (Physarum, DLA, Agent-based, Primordial, Stigmergy)
- RGB tint controls

**Last phase:** Phase 2 (completed)
