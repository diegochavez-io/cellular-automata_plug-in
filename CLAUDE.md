# DayDream Scope - Krea Realtime Video Pipeline Project

## Quick Start (RunPod)

**SSH Connection:**
```bash
ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io
```

**Key Paths on RunPod:**
- **LoRAs**: `/workspace/models/lora/` (NOT `~/.daydream-scope`)
- **Logs**: `/workspace/logs/`
- **Scope Config**: Check `env | grep SCOPE`

**Check LoRAs:**
```bash
ls -lh /workspace/models/lora/
```

**View Logs:**
```bash
tail -50 /workspace/logs/*.log
```

**Transfer Files (using croc):**
1. Local: `env CROC_SECRET="runpod-loras-2024" croc send *.safetensors &`
2. RunPod: `cd /workspace/models/lora/ && CROC_SECRET="runpod-loras-2024" croc --yes --overwrite`

---

## Project Overview
This project focuses on running the Krea Realtime Video Pipeline on RunPod with custom 14B LoRAs. We build resources locally and deploy via SSH to RunPod for GPU-accelerated video generation.

## Technical Stack

### Container & Deployment
- **Container Image**: `daydreamlive/scope:main`
- **Platform**: RunPod (GPU cloud)
- **Deployment Method**: Build locally → Deploy via SSH to RunPod
- **HuggingFace Token**: `$HF_TOKEN`

### Model & Training
- **Pipeline**: Krea Realtime (Wan2.1-T2V-14B)
- **LoRA Training**: 14B WAN LoRAs (custom training in progress)
- **LoRA Format**: `.safetensors` files
- **Model Directory**: `/workspace/models/lora` (set via `SCOPE_LORA_DIR` environment variable)

### MCP Server Integration
The project uses Context7 MCP servers for real-time documentation access:

**DayDream Live Documentation**
- **Source**: https://context7.com/llmstxt/daydream_live_llms-full_txt/llms.txt?tokens=10000
- **Coverage**: LoRAs, plugins, pipelines, Scope API

**RunPod Documentation**
- **Source**: https://context7.com/websites/runpod_io/llms.txt?tokens=10000
- **Coverage**: GPU pricing, serverless deployment, Docker, SSH/SCP

**API Key**: `$CTX7_API_KEY`
- Configured in `~/.claude/mcp_settings.json`
- Access both documentation sources
- Restart Claude Code to activate

## RunPod SSH Connection

### Connection Methods
RunPod provides two SSH connection approaches:

#### 1. Gateway SSH (ssh.runpod.io)
**Connection String**: `ssh k48mr1qqbsotow-64411ea8@ssh.runpod.io -i ~/.ssh/id_ed25519`

**Characteristics**:
- Works for all pods (no public IP required)
- **Interactive sessions only** - does not support direct command execution
- No SCP/SFTP support
- PTY limitations when used non-interactively

**Usage from Claude Code**:
```bash
# For interactive sessions with command execution via stdin:
(echo 'commands here && exit') | ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io

# This creates a background session that executes commands and exits
```

#### 2. Direct SSH (Public IP + Port)
**Connection String**: `ssh root@80.15.7.37 -p 39416 -i ~/.ssh/id_ed25519`

**Characteristics**:
- Requires pod with public IP and exposed SSH port
- Full SSH functionality (commands, SCP, SFTP)
- Better for automation and file transfers
- May be unavailable if pod is stopped or ports change

**How to Get Direct Connection Details**:
1. Open RunPod web interface
2. Find pod (ID: `k48mr1qqbsotow-64411ea8`)
3. Click "Connect" button
4. Look for TCP Port Mappings → SSH (port 22)
5. Note the public IP and mapped port

### Claude Code SSH Workflow

When connecting to RunPod from Claude Code:

1. **Use `-tt` flag** to force TTY allocation for gateway connections
2. **Pipe commands via stdin** to execute in the session:
   ```bash
   (echo 'command1 && command2 && exit') | ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io
   ```
3. **Check background task output** - SSH commands run as background tasks:
   ```bash
   # Command runs with task ID like: b796c85
   # Read output from: /private/tmp/claude-501/-Users-agi-Code-daydream-scope/tasks/[task_id].output
   ```
4. **Prefer direct SSH** when available for cleaner command execution

### Common SSH Tasks

#### Check Pod Status
```bash
(echo 'hostname && uptime && nvidia-smi && docker ps && exit') | \
  ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io
```

#### Deploy LoRAs

**Method 1: Using croc (Recommended for gateway-only access)**

Croc is a peer-to-peer file transfer tool that works through NAT/firewalls - perfect for RunPod gateway connections.

```bash
# Step 1: Install croc locally (if not installed)
brew install croc

# Step 2: Install croc on RunPod (one-time setup)
(cat << 'SCRIPT'
echo "Installing croc..."
curl https://getcroc.schollz.com | bash
which croc
exit
SCRIPT
) | ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io

# Step 3: Start sending files from local machine (background process)
cd /Users/agi/Code/daydream_scope/loras
env CROC_SECRET="runpod-loras-2024" croc send *.safetensors > /tmp/croc_send.log 2>&1 &

# Step 4: Start receiving on RunPod (run immediately after starting send)
(cat << 'SCRIPT'
cd /workspace/models/lora/
echo "Receiving files to /workspace/models/lora..."
CROC_SECRET="runpod-loras-2024" croc --yes --overwrite
ls -lh
exit
SCRIPT
) | ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io

# Monitor progress:
# - Local: tail -f /tmp/croc_send.log
# - RunPod: Check the SSH output for progress bars
```

**Key Points:**
- Use the SAME `CROC_SECRET` value on both sides
- Start the send process first (in background with `&`)
- Start the receive process within a few seconds
- Croc handles encryption, compression, and resumption automatically
- Transfer speed: ~1.5-2.0 MB/s through gateway

**Method 2: Via direct SSH with SCP (when port 22 is exposed)**

```bash
# Only works if direct SSH connection is available
scp -P 39416 -i ~/.ssh/id_ed25519 ./lora-model.safetensors \
  root@80.15.7.37:~/.daydream-scope/models/lora/
```

**Method 3: Via wget on RunPod (for public URLs)**

```bash
# If LoRAs are hosted online (HuggingFace, CivitAI, etc.)
(echo 'wget https://url/to/lora.safetensors -P ~/.daydream-scope/models/lora/ && exit') | \
  ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io
```

#### Start Scope Container
```bash
(cat << 'SCRIPT'
export HF_TOKEN=$HF_TOKEN
docker pull daydreamlive/scope:main
docker run -d \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.daydream-scope/models:/root/.daydream-scope/models \
  -p 8000:8000 \
  --name scope-main \
  daydreamlive/scope:main
docker ps
exit
SCRIPT
) | ssh -tt -i ~/.ssh/id_ed25519 k48mr1qqbsotow-64411ea8@ssh.runpod.io
```

### Troubleshooting SSH

**Error: "Your SSH client doesn't support PTY"**
- This appears when using gateway without `-tt` flag
- Solution: Add `-tt` flag and pipe commands via stdin

**Error: "Connection refused" (Direct SSH)**
- Pod may be stopped - check RunPod web interface
- SSH port may not be exposed - verify TCP port mappings
- IP/port may have changed - get fresh connection details

**Commands not executing**
- Ensure commands end with `&& exit` to close session
- Use parentheses with echo to pipe entire command block
- Check background task output files for results

## Architecture

### Pipeline Compatibility Matrix
| Pipeline | Model Size | LoRA Version Required |
|----------|------------|----------------------|
| Krea Realtime | 14B | Wan2.1-T2V-14B |
| StreamDiffusion V2, LongLive, RewardForcing, MemFlow | 1.3B | Wan2.1-T2V-1.3B |

### Recommended 14B LoRAs
- **Origami**: Paper-craft aesthetic
- **Film Noir**: Classic cinema style
- **Pixar**: 3D animation style

## Development Workflow

### 1. Local Development
```bash
# Install Scope locally
uv run daydream-scope install <source>

# Test plugin development
# - Modify code
# - Reload plugin in Scope interface
# - Verify functionality
```

### 2. LoRA Management
```bash
# Local installation
# Download .safetensors files to: ~/.daydream-scope/models/lora

# Cloud deployment (RunPod)
# IMPORTANT: On RunPod, LoRAs must be in /workspace/models/lora (not ~/.daydream-scope)
wget <lora-url> -P /workspace/models/lora
# For CivitAI: Add API key authentication token
```

### 3. Build & Deploy to RunPod
```bash
# Local build
docker build -t daydream-scope-local .

# Deploy via SSH to RunPod
ssh root@<runpod-instance> << 'EOF'
  # Pull latest container
  docker pull daydreamlive/scope:main

  # Set HuggingFace token
  export HF_TOKEN=$HF_TOKEN

  # Run container with LoRA mounting
  docker run -d \
    -e HF_TOKEN=$HF_TOKEN \
    -v ~/.daydream-scope/models:/root/.daydream-scope/models \
    -p 8000:8000 \
    daydreamlive/scope:main
EOF
```

## Plugin Development

### Core Concepts
Scope plugins extend functionality through a hook-based system. Plugins can add:
- Custom pipelines
- Preprocessors
- Postprocessors

### Plugin Registration (`pyproject.toml`)
```toml
[project.entry-points."scope"]
my_scope_plugin = "my_scope_plugin.plugin"
```

### Pipeline Types

#### Text-Only Pipelines
- Generate content without video input
- No `prepare()` method required
- Tensor format: THWC with values in [0, 1] range
- Read runtime parameters from `kwargs` in `__call__()`

#### Video Input Pipelines
- Process incoming frames
- Implement `prepare()` returning `Requirements(input_size=N)`
- Input: [0, 255], Output must be: [0, 1]
- Access frames via `kwargs.get("video")`

### Parameter Strategy

| Type | Timing | Access Point | Notes |
|------|--------|--------------|-------|
| Load-time | Initialization | `__init__()` | Set once at pipeline load |
| Runtime | Per-frame | `__call__()` kwargs | Updated dynamically |

**Critical**: Runtime parameters MUST be read from `kwargs` in `__call__()`, not stored in `__init__()`

### UI Configuration
Use `ui_field_config()` to:
- Control parameter ordering
- Set mode-specific visibility
- Categorize into Settings or Input & Controls panels

## LoRA Configuration

### Runtime Mode (Dynamic Scaling)
```python
# Enable dynamic LoRA scaling without pipeline restart
pipeline = load_pipeline(
    lora_merge_mode="runtime_peft"
)
```

**Note**: Runtime mode allows swapping LoRAs on-the-fly but may slightly reduce performance compared to permanent merging.

### Multiple LoRAs
- Load and apply multiple LoRAs simultaneously
- Adjust individual LoRA influence scales
- Combine styles (e.g., Origami + Film Noir)

## File Structure
```
daydream_scope/
├── CLAUDE.md                    # This file
├── plugins/                     # Custom Scope plugins
│   └── my_plugin/
│       ├── __init__.py
│       ├── plugin.py           # Plugin registration
│       └── pipeline.py         # Pipeline implementation
├── loras/                      # Local LoRA training/testing
│   └── *.safetensors          # Trained 14B LoRAs
├── docker/
│   └── Dockerfile             # Custom container builds
└── deploy/
    └── runpod_deploy.sh       # SSH deployment script
```

## Key Documentation Resources
- **LoRA Guide**: https://docs.daydream.live/scope/guides/loras
- **Plugin Development**: https://docs.daydream.live/scope/guides/plugin-development
- **MCP Server Docs**: https://context7.com/llmstxt/daydream_live_llms-full_txt/llms.txt?tokens=10000

## Development Guidelines

### When Adding New Pipelines
1. Create plugin structure with proper entry points
2. Implement required hooks (`register_pipelines`, `prepare()` for video inputs)
3. Test locally with Scope interface
4. Build container and deploy to RunPod
5. Verify GPU acceleration and LoRA loading

### When Training New LoRAs
1. Ensure compatibility with Wan2.1-T2V-14B architecture
2. Save as `.safetensors` format
3. Test locally in `~/.daydream-scope/models/lora`
4. Upload to HuggingFace or CivitAI for cloud deployment
5. Update deployment scripts with new LoRA URLs

### When Deploying to RunPod
1. Test container locally first
2. Verify HF_TOKEN is set correctly
3. Ensure LoRA models are accessible in container
4. Monitor GPU memory usage (14B models are large)
5. Test video pipeline latency and quality

## Environment Variables

### RunPod Container Environment
When running on RunPod, these environment variables are automatically set:
```bash
# Scope configuration (READ-ONLY - set by container)
DAYDREAM_SCOPE_LOGS_DIR=/workspace/logs
DAYDREAM_SCOPE_MODELS_DIR=/workspace/models
SCOPE_LORA_DIR=/workspace/models/lora  # LoRAs MUST be here on RunPod!

# HuggingFace token (set by you)
HF_TOKEN=$HF_TOKEN
```

### Local Development
```bash
# Required for HuggingFace model access
export HF_TOKEN=$HF_TOKEN

# Optional: Override LoRA directory (local only)
export SCOPE_LORA_DIR=~/.daydream-scope/models/lora

# Optional: Set GPU device
export CUDA_VISIBLE_DEVICES=0
```

**IMPORTANT**: On RunPod, always use `/workspace/models/lora` for LoRAs, NOT the home directory path!

## Common Tasks

### Load Pipeline with Custom LoRA
```python
from scope import load_pipeline

pipeline = load_pipeline(
    "krea_realtime",
    loras=["origami", "film_noir"],
    lora_scales=[0.8, 0.5],
    lora_merge_mode="runtime_peft"
)
```

### Reload Plugin During Development
```bash
# In Scope interface
# 1. Modify plugin code
# 2. Click "Reload Plugin"
# 3. Test changes immediately
```

### Monitor RunPod Container
```bash
# SSH into RunPod instance
ssh root@<runpod-instance>

# Check container logs
docker logs -f <container-id>

# Monitor GPU usage
nvidia-smi -l 1
```

## Troubleshooting

### LoRA Not Loading or "Pipeline Error"
- **CRITICAL**: On RunPod, LoRAs MUST be in `/workspace/models/lora/` (not `~/.daydream-scope/models/lora/`)
- Check environment variable: `SCOPE_LORA_DIR=/workspace/models/lora`
- Verify `.safetensors` files are not corrupted (check file size ~147MB for 14B models)
- If you see "Error while deserializing header: invalid JSON", the file is corrupted - delete and re-transfer
- Check LoRA is 14B version (Wan2.1-T2V-14B) for Krea Realtime
- Ensure file permissions allow read access: `chmod 644 /workspace/models/lora/*.safetensors`

### Corrupted Files After Transfer
- If croc transfer is interrupted, files may be corrupted
- Symptoms: "invalid JSON in header" or "control character" errors
- Solution: Delete corrupted file and re-transfer
- Verify file integrity: Check file size matches expected size

### SSH Connection Issues
- **Gateway method** (`ssh.runpod.io`): Use `-tt` flag for interactive sessions
- **Direct SSH refused**: Pod may be stopped, check RunPod web interface
- **PTY errors**: Normal with gateway, use the documented `-tt` method
- **File transfer fails**: Use croc (not SCP) with gateway connections

### Container Fails to Start
- Verify HF_TOKEN is set correctly
- Check GPU availability on RunPod instance
- Review container logs: `tail /workspace/logs/*.log`

### Pipeline Performance Issues
- Consider using permanent LoRA merge instead of runtime mode
- Reduce number of simultaneous LoRAs
- Lower LoRA scales to reduce computational overhead
- Check GPU memory: `nvidia-smi`

## Cellular Automata Plugin — Video Source for Scope

### What It Is
A living cellular automata organism that generates constant-motion video as input for Scope's Krea Realtime pipeline. The CA output feeds into the AI video pipeline with LoRAs for final aesthetic transformation. Think: microscopic bioluminescent organism → AI-enhanced art video.

### Current State (Local Prototype — Final Tweaks)
- **Location**: `plugins/cellular_automata/`
- **Run**: `cd plugins && python3 -m cellular_automata coral`
- **GitHub**: `diegochavez-io/cellular-automata_plug-in`
- **Stack**: pygame-ce, numpy, scipy
- **Resolution**: 1024x1024 (Lenia/SL/MNCA), 512x512 (GS)
- **FPS**: Lenia ~12-18, SmoothLife ~8-12, MNCA ~10-15, GS ~30-55

**Working engines**: Lenia (8 presets), SmoothLife (5 presets), MNCA (5 presets), Gray-Scott (5 presets)

**Key features built**:
- Universal flow fields: 7 types (radial, rotate, swirl, bubble, ring, vortex, vertical), 0.8px/step semi-Lagrangian advection, all presets ship with 5 active flow channels
- Vivid multi-zone color: full-gamut cosine palettes with 2nd+3rd harmonics, spatial noise blobs create distinct color zones across organism
- Float32 throughout all engines for ~2x speedup
- Pre-computed noise pool (6 full-res frames, zero per-frame cost)
- Thickness slider for line dilation control
- GS contrast stretch (V*3.0), SmoothLife contrast curve (world^1.8)
- Rotating stir current prevents center crystallization
- Velocity-driven perturbation: static pixels always get pushed
- Coverage bounds (25%-85%) with noise-based management
- LFO breathing modulation per engine (Lenia: mu/sigma/T, SL: b1/b2, MNCA: delta, GS: feed)
- Radial containment keeps organism centered on black background
- Auto-reseed on death (no flash)

**Status**: Doing small visual tweaks (preset tuning, color refinement). Core systems are solid. Next big step is wrapping as a Scope plugin (Phase 2).

### Deployment Roadmap

**Phase 1: Finalize Local Prototype** ← ALMOST DONE (small tweaks remaining)
- [x] Core engines working (Lenia, SmoothLife, MNCA, Gray-Scott)
- [x] Universal flow field system (7 types, all presets cranked)
- [x] Vivid multi-zone color pipeline (harmonics, spatial zones)
- [x] Organic movement (rotating stir, velocity perturbation)
- [x] Coverage bounds and auto-reseed
- [x] Performance optimization (float32, noise pool, LUT caching)
- [x] Thickness slider
- [ ] Final preset tuning (user cycling through giving feedback)
- [ ] Color refinement (closer to bioluminescent reference images)

**Phase 2: Scope Plugin Wrapper**
- [ ] Create text-only pipeline class (output THWC tensors in [0,1])
- [ ] Implement `register_pipelines` hook in plugin.py
- [ ] Map CA controls to Scope runtime `kwargs` parameters
- [ ] Add `ui_field_config()` for Scope UI (preset selector, speed, hue)
- [ ] Test plugin structure locally with `uv run daydream-scope install`

**Phase 3: GPU Acceleration (CuPy)**
- [ ] Port numpy → CuPy (drop-in GPU replacement)
- [ ] Port scipy.ndimage → cupyx.scipy.ndimage (gaussian_filter, etc.)
- [ ] Add numpy/cupy toggle based on GPU availability
- [ ] Target: 30+ FPS at 1024x1024 on RTX 5090

**Phase 4: RunPod Deployment**
- [ ] Spin up RunPod pod with RTX 5090
- [ ] Install CuPy + dependencies in container
- [ ] Deploy CA plugin to Scope
- [ ] Test full pipeline: CA → Scope → Krea Realtime + LoRAs
- [ ] Tune CA parameters for best AI video input quality
- [ ] Performance profiling and optimization

### Key Technical Decisions
- **No web intermediate** — go direct to Scope plugin (code is already Python)
- **CuPy over CUDA kernels** — nearly identical API to numpy, minimal rewrite
- **Text-only pipeline** — CA generates frames, no video input needed
- **Runtime params via kwargs** — preset, speed, hue controlled from Scope UI

### Key Files
| File | Purpose |
|------|---------|
| `viewer.py` | Main loop, sim stepping, containment, perturbation, rendering |
| `iridescent.py` | Cosine palette color pipeline, 2D LUT, bloom |
| `lenia.py` | Lenia engine (FFT convolution, growth function) |
| `smoothlife.py` | SmoothLife engine (continuous GoL, sigmoid transitions) |
| `mnca.py` | Multi-Neighborhood CA engine (ring kernels, threshold rules) |
| `presets.py` | All preset definitions and UNIFIED_ORDER |
| `controls.py` | pygame UI widgets (sliders, buttons, panel) |
| `smoothing.py` | EMA parameter smoothing and LFO systems |
| `engine_base.py` | Abstract base class for all CA engines |

---

## Session Status
1. ✅ Set up MCP server connection for documentation access
2. ✅ Established SSH connection to RunPod (gateway method documented)
3. ✅ Set up croc for file transfer through SSH gateway
4. ✅ Transferred 7+ working LoRAs to `/workspace/models/lora/`
5. ✅ Tested LoRAs with Krea Realtime pipeline on RTX 5090
6. ✅ Documented correct paths and troubleshooting steps
7. 🔄 Complete remaining 14B LoRA training
8. ⬜ Build custom deployment container
9. ✅ CA prototype working locally — all engines, flow, color, performance
10. 🔄 Final CA visual tweaks (small — preset tuning, color refinement)
11. ⬜ Write Scope plugin wrapper for CA (Phase 2 — next big step)
12. ⬜ Port CA to CuPy for GPU acceleration
13. ⬜ Deploy CA plugin to RunPod and test full pipeline

## Important Notes for Next Session
- **RunPod Pod ID**: `k48mr1qqbsotow-64411ea8`
- **SSH Gateway**: `ssh.runpod.io` (use `-tt` flag)
- **Direct SSH** (when available): `root@80.15.7.37 -p 39416`
- **LoRAs Location**: `/workspace/models/lora/` (7 working files transferred)
- **Container**: Already running Scope on port 8000
- **GPU**: RTX 5090, 32GB VRAM
- **CA Plugin**: Run with `cd plugins && python3 -m cellular_automata coral`
- **CA Status**: Small visual tweaks remaining, then Phase 2 (Scope plugin wrapper) — a huge undertaking
- **CA Latest Commit**: `ea03e2a` — universal flow, vivid color, performance
- **Key CA decisions locked**: Flow at 0.8px/step, full-gamut palettes w/ harmonics, spatial noise zones, float32, pre-computed noise pool
- **Known Issues**: One corrupted LoRA removed (`styl-gn_2602_000000759.safetensors`)
