# Phase 2: Iridescent Color Pipeline - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace the existing 4-layer HSV color system (Core/Halo/Spark/Memory) with a unified iridescent rendering pipeline. All engines render through the same pipeline. The output is a cuttlefish/bioluminescent aesthetic with multi-channel mathematical color mapping. No new engines or presets — just the color pipeline swap.

</domain>

<decisions>
## Implementation Decisions

### Shimmer character
- **Gentle flow speed** — visible color drift like ink in water, ~10-15s for a noticeable shift across the surface
- **Complementary hue shift** between interior and edges — same color family but shifted (e.g. teal interior bleeding into blue-violet edges). Not jarring contrast, not identical — related but distinct
- **Mix of saturation** — dense areas are rich and saturated, thin/edge areas glow translucent. The organism's own structure determines whether you get bold or ethereal
- **Bioluminescent fine detail** — bright specks/particles at edges and high-gradient regions for living texture

### Spatial color mapping
- **Multi-channel mathematical mapping** — NOT a simple hue sweep. Different simulation properties (density, gradient magnitude, rate of change) drive different color channels. The organic feel comes from the math
- **Structure-following distribution** — colors follow the organism's own topology. Dense clusters, folds, boundaries, and tendrils each get naturally different colors based on local simulation state
- **Subtle ambient glow** around the organism — soft haze like underwater bioluminescence, organism gently lights its surroundings before falling to black

### Hue sweep & animation
- **Tied to LFO breathing** — color animation synchronized with the LFO cycle, not independent
- **Hue rotation on each breath cycle** — each LFO breath advances the global hue by a step. Over many breaths, the organism cycles through the full rainbow
- **Full rainbow cycle speed** — Claude's discretion (pick a speed that creates constantly-fresh visuals without feeling rushed)
- **Per-preset tint offset** — Claude's discretion on whether presets have different default color biases to feel visually distinct

### Slider controls
- **RGB tint sliders** — Claude's discretion on implementation (color balance shift vs channel intensity — whichever gives best creative control without breaking the iridescent look)
- **Brightness slider uses exposure/gamma** — non-linear control that brings out detail in dark areas without blowing out highlights. Cinematic feel
- **Default slider position** — Claude's discretion (should look great out of the box, matching the cuttlefish/bioluminescent aesthetic)
- **Double-click to reset** any slider to its default value

### Claude's Discretion
- Caustic/depth approach (density-driven vs animated noise layer vs both — pick whichever gives the most organic cuttlefish feel)
- Full rainbow cycle duration (balance freshness vs naturalness)
- Per-preset color tint offsets (whether and how presets differ visually)
- RGB tint slider implementation approach
- Default slider positions for out-of-box look
- Technical color math (which simulation channels map to which color properties)

</decisions>

<specifics>
## Specific Ideas

- **Cuttlefish as primary visual reference** — the organism should feel alive, bioluminescent, like a deep-sea creature
- **Reference image analysis**: Dense interior = cyan/teal, fold boundaries = orange/warm, outer edges = deep blue, transition zones = green. Multiple simultaneous hues, never monochromatic
- **Translucent caustic depth** — light should feel like it passes through the form, not painted on the surface
- Colors should have "balance and flow" — complex but harmonious, not chaotic
- Interior vs edges must have different hues — this is essential to the aesthetic
- Fine bioluminescent particles/specks at edges add living texture

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-iridescent-color-pipeline*
*Context gathered: 2026-02-16*
