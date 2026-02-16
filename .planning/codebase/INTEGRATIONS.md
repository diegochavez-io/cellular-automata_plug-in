# External Integrations

**Analysis Date:** 2026-02-16

## APIs & External Services

**Hugging Face Model Hub:**
- Service - Model downloading, LoRA management, and model hosting
  - SDK/Client: `transformers`, `huggingface_hub` (imported implicitly by Scope)
  - Auth: Environment variable `HF_TOKEN=HF_TOKEN_REDACTED`
  - Usage: Loads Wan2.1-T2V-14B model and custom LoRAs

**Context7 MCP Documentation API:**
- Service - Real-time documentation access for DayDream Live and RunPod
  - Endpoint: `https://context7.com/llmstxt/`
  - API Key: `ctx7sk-77bb38a0-d485-4125-ad2d-04183138ebef`
  - Coverage: LoRA guides, plugin development docs, Scope API reference
  - Configuration: Stored in `~/.claude/mcp_settings.json`

**CivitAI (Model Hub):**
- Service - Public LoRA model hosting and discovery
  - Authentication: API key when needed (mentioned in CLAUDE.md)
  - Usage: Alternative source for downloading trained LoRAs

## Data Storage

**Local File Storage:**
- LoRA Models: `~/.daydream-scope/models/lora/` (local dev), `/workspace/models/lora/` (RunPod)
  - Format: `.safetensors` (binary tensor format)
  - Size: ~147MB per 14B LoRA model
  - Lifecycle: Persisted across sessions, synced via croc

**Training Datasets:**
- Path: `/app/ai-toolkit/datasets/` (within training container)
- Format: Video frames + text captions
  - Example: `datasets/slm_mld_grth/` - Slime mold growth training data
  - Resolution: 480x832 (from `configs/slm_mld_grth_config.yaml`)
  - Num frames: 80 frames per video sample

**Training Database:**
- SQLite - AI Toolkit database
  - Path: `/app/ai-toolkit/aitk_db.db`
  - Purpose: Tracks training jobs, datasets, and model saves

**RunPod Network Volume (Optional):**
- S3-Compatible API endpoint
  - Pattern: `https://<volume-id>.runpod.io`
  - Authentication: AWS-style access keys
  - Usage: Store trained LoRAs and training outputs (not currently in use)

## Cache & Temporary Storage

**Model Cache:**
- HuggingFace cache directory (auto-managed by transformers)
- Hugging Face models cached during first load
- LoRA safetensors loaded into memory during pipeline initialization

**Training Outputs:**
- Directory: `/app/ai-toolkit/output/`
- Format: Diffusers format checkpoint saves
- Frequency: Every 200 steps (configurable via `save_every` in training config)
- Retention: Max 6 saves per training job (via `max_step_saves_to_keep`)

## Authentication & Identity

**HuggingFace Token:**
- Type: API token for model hub access
- Storage: Environment variable `HF_TOKEN`
- Scope: Required for downloading Wan2.1-T2V-14B model and reading private LoRA repos
- Lifespan: Long-lived, stored in container environment and `.env`

**SSH Private Key:**
- Type: Ed25519 key pair
- Storage: `~/.ssh/id_ed25519` (local machine)
- Usage: Authenticating to RunPod instances via SSH gateway
- Connection methods:
  - Gateway: `ssh.runpod.io` (requires `-tt` flag)
  - Direct: `root@<pod-ip> -p <mapped-port>`

**RunPod API (Optional):**
- Type: API key for programmatic pod management
- Env var: `RUNPOD_API_KEY` (defined in `.env.example`, not currently used)
- Purpose: Would enable automated pod creation/teardown

## Monitoring & Observability

**Error Tracking:**
- Not detected - Errors logged to container stdout

**Logs:**
- Container logs: `docker logs daydream-scope` on RunPod
- Path: `/workspace/logs/` on RunPod (via `DAYDREAM_SCOPE_LOGS_DIR`)
- Training logs: Accessible via AI Toolkit UI logger
- Deployment script monitors: GPU status via `nvidia-smi`

**Metrics Collection:**
- Training: Performance logged every 10 steps to SQLite
- GPU monitoring: `nvidia-smi` for VRAM and utilization
- No external metrics service integration

## CI/CD & Deployment

**Hosting:**
- RunPod - GPU cloud platform
  - Pod ID: `k48mr1qqbsotow-64411ea8`
  - Gateway SSH: `ssh.runpod.io`
  - Direct SSH: `root@80.15.7.37 -p 39416` (when pod has public IP)

**Container Registry:**
- Docker Hub - daydreamlive/scope:main
- Image source: `docker pull daydreamlive/scope:main`
- Deployment: Pulled on RunPod instance during deployment

**Deployment Mechanism:**
- SSH-based (no CI/CD pipeline)
- Script: `deploy/runpod_deploy.sh`
- Manual trigger via CLI with RunPod host parameter
- File transfer: `croc` for P2P LoRA transfers through SSH gateway

**SSH Gateway:**
- Type: RunPod SSH gateway
- Endpoint: `ssh.runpod.io`
- Purpose: Access to pods without public IPs
- Limitations: Interactive sessions only, no SCP/SFTP direct support
- Workaround: Use croc for file transfers

## Environment Configuration

**Required Environment Variables:**
- `HF_TOKEN` - HuggingFace API token (critical for model downloads)
- `CUDA_VISIBLE_DEVICES` - GPU selection (0 for first GPU)
- `SCOPE_LORA_DIR` - Directory path to LoRA models

**RunPod Container Environment (Auto-set):**
- `DAYDREAM_SCOPE_LOGS_DIR=/workspace/logs`
- `DAYDREAM_SCOPE_MODELS_DIR=/workspace/models`
- `SCOPE_LORA_DIR=/workspace/models/lora` (CRITICAL: must be /workspace on RunPod, not home dir)

**Training Environment:**
- `DAYDREAM_SCOPE_LOGS_DIR` - Logs for training runs
- Database connection - Sqlite at `/app/ai-toolkit/aitk_db.db`

**Secrets Location:**
- `.env` file (local, git-ignored)
- Container environment variables (set in `docker run -e`)
- SSH keys: `~/.ssh/id_ed25519`
- MCP API key: `~/.claude/mcp_settings.json`

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook receivers in codebase

**Outgoing:**
- Not detected - No webhook senders configured

**Training Events:**
- Training sampler outputs generated every 200 steps
- Sampling process defined in config: 4 prompts sampled with FPS 16, 80 frames per sample
- Outputs saved as Diffusers checkpoint format

## File Transfer

**Croc (P2P File Transfer):**
- Purpose: Transfer LoRA models through SSH gateway
- Protocol: End-to-end encrypted, works through NAT/firewalls
- Usage pattern:
  1. Local: `croc send *.safetensors` (background)
  2. RunPod: `croc --yes --overwrite` (receives in `/workspace/models/lora/`)
- Secret: `CROC_SECRET=runpod-loras-2024`
- Speed: ~1.5-2.0 MB/s through gateway

**Deployment Scripts:**
- `deploy/runpod_deploy.sh` - Full container deployment via SSH
- `deploy/sync_loras.sh` - LoRA file synchronization script

---

*Integration audit: 2026-02-16*
