# Phase 3: Safe Parameter Control - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

EMA smoothing on parameter sliders prevents organism death from abrupt changes. Parameter coupling keeps mu/sigma in balance. The creature never visibly dies or restarts. This phase does NOT add new presets, engines, or UI layout changes.

</domain>

<decisions>
## Implementation Decisions

### Parameter coupling (mu/sigma dance)
- mu and sigma should communicate — when one moves, the other adjusts to keep the organism in a viable zone
- This is a "delicate dance" — parameters cooperate to maintain life, not just clamp independently
- User wants both a high-level "character" slider (calm <-> energetic) AND advanced mu/sigma sliders for fine-tuning
- Character slider moves mu/sigma together behind the scenes; advanced sliders still cooperate internally

### Slider response feel
- Dreamy, 2-3 second transitions — parameters drift slowly to new values
- Changes should feel organic, like the creature is lazily morphing to new behavior
- No jarring jumps, ever

### Survival priority
- Life above all — the creature NEVER dies, even if that means overriding extreme slider positions
- If the organism starts fading, parameters silently drift back toward safe territory (invisible rescue)
- No visual indicator of rescue — it should look like the creature is just resilient
- Recovery from near-death should be seamless, never showing a restart

### Preset transitions
- Switching presets (number keys 1-9) should smooth-morph parameters over 2-3 seconds
- The creature transforms into the new preset's behavior rather than resetting
- No reseed on preset switch — the living organism evolves into the new form

### Claude's Discretion
- Coupling approach: how exactly mu/sigma communicate (ratio-based, lookup table, constraint zone — Claude picks what keeps creature alive best)
- Death recovery mechanism: fade-in regrowth vs ghost persistence vs other approach — whatever looks most organic
- Containment mask role: whether it actively injects density to prevent death or stays passive — Claude decides
- EMA time constants: exact smoothing values within the 2-3 second feel
- Safe bounds per engine: specific clamping ranges for each engine type
- Which parameters beyond mu/sigma need smoothing (kernel radius, T, etc.)

</decisions>

<specifics>
## Specific Ideas

- "Like a delicate dance" — mu and sigma balancing the creature's movement and floating
- "Exciting video input for DayDream Scope" — this is a living video source, constant movement is the priority
- "Organism, like creature floating around the middle" — centered, alive, undulating
- "I don't want to see the restart" — death/rebirth must be invisible if it happens at all
- User said "life above all" — survival trumps slider accuracy at extremes

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-safe-parameter-control*
*Context gathered: 2026-02-16*
