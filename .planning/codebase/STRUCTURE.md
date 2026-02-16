# Codebase Structure

**Analysis Date:** 2026-02-16

## Directory Layout

```
daydream_scope/
├── CLAUDE.md                           # Project context for Claude Code
├── QUICK_START_GUIDE.md               # User quick-start guide
├── SCOPE_QUICK_REF.md                 # Quick reference for Scope deployment
├── LORA_TRAINING_*.md                 # LoRA training session logs
├── README.md                          # Project overview
├── MCP_SETUP.md                       # Context7 MCP server configuration
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore patterns
│
├── plugins/                           # DayDream Scope plugins
│   ├── cellular_automata/            # Cellular Automata plugin (primary work)
│   │   ├── __init__.py               # Package marker
│   │   ├── __main__.py               # Standalone viewer entry point
│   │   ├── plugin.py                 # Scope plugin registration hook
│   │   │
│   │   ├── viewer.py                 # Main interactive loop & UI orchestration
│   │   ├── controls.py               # Pygame UI widgets (Slider, Button, ControlPanel)
│   │   ├── color_layers.py           # 4-layer rendering & feedback system
│   │   ├── presets.py                # Engine parameter definitions & lookup
│   │   ├── colormaps.py              # Color utility functions (HSV, etc.)
│   │   │
│   │   ├── engine_base.py            # Abstract CAEngine base class
│   │   ├── lenia.py                  # Lenia continuous CA with FFT
│   │   ├── life.py                   # Game of Life variants (B/S rules)
│   │   ├── excitable.py              # Greenberg-Hastings excitable media
│   │   └── gray_scott.py             # Gray-Scott reaction-diffusion
│   │
│   └── example_plugin/               # Template for new plugins
│       ├── __init__.py               # Package marker
│       ├── plugin.py                 # Scope registration hook (template)
│       └── pipeline.py               # Pipeline implementations (3 patterns shown)
│
├── loras/                            # Trained 14B LoRA models
│   └── *.safetensors                # Individual LoRA files (deployed to RunPod)
│
├── configs/                          # Configuration files (if any)
│
├── deploy/                          # Deployment utilities
│   ├── runpod_deploy.sh            # SSH deployment script for RunPod
│   ├── sync_loras.sh               # LoRA file sync script
│   └── runpod_info.md              # RunPod connection details & notes
│
├── docker/                         # Container configuration
│   └── [Dockerfile if custom builds needed]
│
├── samples/                        # Sample outputs/logs
│   └── slm_mld_grth_v2/           # Training session output directory
│
├── blog_assets/                    # Documentation assets
│   ├── anmdf_mlt/
│   ├── slm_mld_grth/
│   └── tdcrtv/
│
├── screenshots/                    # GUI screenshots
│
├── timeline_json/                  # Timeline/session tracking
│
└── .planning/codebase/            # GSD analysis documents (this directory)
    ├── ARCHITECTURE.md
    └── STRUCTURE.md
```

## Directory Purposes

**plugins/cellular_automata/:**
- Purpose: Complete standalone CA visualization system + Scope plugin integration point
- Contains: Engine implementations, rendering system, interactive viewer, UI controls, presets
- Key files: `viewer.py` (orchestration), `engine_base.py` (interface), engine implementations

**plugins/example_plugin/:**
- Purpose: Template for developers creating new Scope plugins
- Contains: Three canonical pipeline patterns (dual-input, text-only, video-only)
- Key files: `plugin.py` (Scope registration hook), `pipeline.py` (implementation examples)

**loras/:**
- Purpose: Storage for trained 14B LoRA files ready for deployment
- Contains: `.safetensors` format model files
- Local path: `~/.daydream-scope/models/lora/`
- RunPod path: `/workspace/models/lora/` (IMPORTANT: Different location than local)

**deploy/:**
- Purpose: SSH scripts and connection documentation for RunPod deployment
- Contains: Shell scripts for deployment automation, connection info
- Key files: `runpod_deploy.sh` (main entry point), `sync_loras.sh` (LoRA transfer), `runpod_info.md` (connection details)

**configs/:**
- Purpose: Configuration files (environment-specific settings, if any)
- Contains: Usually minimal; most config via environment variables

**docker/:**
- Purpose: Custom Docker container definitions
- Contains: Dockerfile for custom image builds (currently empty, uses daydreamlive/scope:main)

**blog_assets/, samples/, screenshots/, timeline_json/:**
- Purpose: Documentation, training logs, and media artifacts
- Usage: Reference for historical context and asset management
- Not critical to active development

## Key File Locations

**Entry Points:**
- `plugins/cellular_automata/__main__.py`: Standalone viewer (run via `python -m cellular_automata`)
- `plugins/cellular_automata/plugin.py`: Scope plugin registration
- `plugins/example_plugin/plugin.py`: Template plugin entry point

**Configuration:**
- `CLAUDE.md`: Project context and development guidelines
- `.env.example`: Environment variable template (copy to .env)
- `plugins/cellular_automata/presets.py`: All CA engine parameter definitions

**Core Logic:**
- `plugins/cellular_automata/viewer.py`: Main event loop, preset switching, LFO modulation
- `plugins/cellular_automata/engine_base.py`: Abstract interface all engines implement
- `plugins/cellular_automata/lenia.py`: FFT-accelerated Lenia engine
- `plugins/cellular_automata/life.py`: Game of Life with B/S rules
- `plugins/cellular_automata/gray_scott.py`: Reaction-diffusion patterns
- `plugins/cellular_automata/excitable.py`: Greenberg-Hastings waves

**Visualization/UI:**
- `plugins/cellular_automata/color_layers.py`: 4-layer rendering and feedback
- `plugins/cellular_automata/controls.py`: Pygame UI widget implementations

**Testing/Examples:**
- `plugins/example_plugin/`: Template for plugin development

## Naming Conventions

**Files:**
- Snake_case for all Python modules: `viewer.py`, `engine_base.py`, `color_layers.py`
- Descriptive names matching primary class/functionality: `lenia.py` → Lenia class, `controls.py` → UI widget classes
- Special files: `__init__.py` (package marker), `__main__.py` (entry point), `plugin.py` (Scope hook)

**Directories:**
- Lowercase with underscores: `cellular_automata/`, `example_plugin/`
- Functional grouping: `plugins/` (all Scope plugins), `deploy/` (deployment scripts), `loras/` (model files)

**Classes:**
- PascalCase: `Viewer`, `CAEngine`, `Slider`, `ControlPanel`, `Lenia`, `Life`, `ColorLayerSystem`

**Functions/Methods:**
- snake_case: `step()`, `set_params()`, `get_params()`, `_build_kernel()`, `_apply_preset()`
- Private helpers prefixed with underscore: `_build_containment()`, `_build_noise_mask()`, `_create_engine()`

**Constants:**
- UPPERCASE: `PRESET_ORDER`, `ENGINE_ORDER`, `PANEL_WIDTH`, `BASE_RES`, `THEME` (dict)

**Parameters:**
- snake_case: `sim_size`, `kernel_peaks`, `brush_radius`, `master_feedback`

## Where to Add New Code

**New CA Engine:**
1. Create `plugins/cellular_automata/my_engine.py`
2. Implement class inheriting from `CAEngine` in `engine_base.py`
3. Implement required methods: `step()`, `set_params()`, `get_params()`, `seed()`, `get_slider_defs()`
4. Add engine class to `ENGINE_CLASSES` dict in `viewer.py`
5. Add engine name to `ENGINE_ORDER` list in `presets.py`
6. Define presets in `presets.py` with `"engine": "my_engine_name"`

**New Preset:**
1. Add entry to `PRESETS` dict in `plugins/cellular_automata/presets.py`
2. Include: `"engine"`, `"name"`, `"description"`, engine-specific parameters, `"seed"` type
3. Add preset key to appropriate list in `PRESET_ORDER` or `PRESET_ORDERS[engine]`

**New Scope Plugin:**
1. Copy `plugins/example_plugin/` to `plugins/my_plugin/`
2. Edit `plugin.py`: Implement `register_pipelines(registry)` hook
3. Edit `pipeline.py`: Implement pipeline class(es) with `__init__()`, `__call__()`, optional `prepare()`, optional `ui_field_config()`
4. Create `pyproject.toml` entry point: `[project.entry-points."scope"] my_plugin = "my_plugin.plugin"`
5. Install locally and test via Scope interface

**New UI Widget:**
1. Add class to `plugins/cellular_automata/controls.py`
2. Implement: `__init__()`, `handle_event(pygame.event)` (returns bool if handled), `draw(surface)`
3. Register in `ControlPanel` widget list

**New Color Layer:**
1. Edit `plugins/cellular_automata/color_layers.py`
2. Add layer definition to `LAYER_DEFS` list: `{"name": "...", "default_weight": 0.X, "hue": Y}`
3. Add feedback coefficient to `_FEEDBACK_COEFFS` array
4. Implement computation in `ColorLayerSystem.process()` method

**Utility Functions:**
- Shared helpers (non-engine-specific): `plugins/cellular_automata/colormaps.py`
- Engine-agnostic visualization: Extend `color_layers.py`
- Scope integration helpers: Extend or create new file in plugin root

## Special Directories

**plugins/cellular_automata/__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic on import)
- Committed: No (in .gitignore)

**deploy/ (scripts, not generated):**
- Purpose: SSH deployment automation
- Generated: No (checked in)
- Committed: Yes

**loras/:**
- Purpose: Training artifacts and deployment files
- Generated: Yes (via training pipeline)
- Committed: No (large binary files, use Git LFS if needed)

**.planning/codebase/ (GSD analysis):**
- Purpose: Architecture and structure documentation for code generation tools
- Generated: Yes (via GSD /map-codebase)
- Committed: Yes (markdown documents)

---

*Structure analysis: 2026-02-16*
