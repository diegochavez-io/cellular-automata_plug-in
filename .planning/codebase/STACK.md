# Technology Stack

**Analysis Date:** 2026-02-16

## Languages

**Primary:**
- Python 3.10+ - Core plugin development and video pipeline infrastructure
- YAML - Training configuration files (`configs/*.yaml`)
- Bash - Deployment scripts (`deploy/runpod_deploy.sh`, `deploy/sync_loras.sh`)

**Secondary:**
- Markdown - Documentation and guides

## Runtime

**Environment:**
- Python 3.10 or higher (specified in `plugins/example_plugin/pyproject.toml`)
- Docker containers for cloud deployment (RunPod)

**Package Manager:**
- pip - Python dependency installation
- uv - Recommended package manager for local development (mentioned in CLAUDE.md)

## Frameworks

**Core Video Pipeline:**
- DayDream Scope (`daydreamlive/scope:main` container) - Realtime video generation framework
- Krea Realtime - T2V-14B video generation pipeline

**ML/Tensor Computing:**
- PyTorch (`>=2.0.0`) - Deep learning framework required in `plugins/example_plugin/pyproject.toml`
- numpy (`>=1.24.0`) - Numerical computing, used across all plugins

**Visualization & UI:**
- pygame (`>=2.5.0`) - Interactive viewer for cellular automata (`plugins/cellular_automata/`)
- pygame-ce - Modern pygame variant (noted in MEMORY.md for cellular automata)
- Gradio - Web UI for Scope (port 7860 exposed in deployment)

**LoRA Training:**
- AI Toolkit - Referenced in `configs/*.yaml` for diffusion trainer
- Diffusers - Model architecture and diffusion sampling

## Key Dependencies

**Critical:**
- torch (>=2.0.0) - Tensor operations and model inference in pipelines
- numpy (>=1.24.0) - Array operations, scientific computing across all engines
- pygame (>=2.5.0) - Real-time rendering for cellular automata visualization
- safetensors - LoRA model format (`.safetensors` file format used throughout project)

**Model Loading:**
- Hugging Face Hub - Model downloads and management
- transformers - Text encoding and model architectures

**Infrastructure:**
- Docker - Container runtime for RunPod deployment
- croc - P2P file transfer (used for LoRA deployment through SSH gateway)

## Configuration

**Environment:**
- HF_TOKEN - HuggingFace authentication token (required for model access)
  - Value: `HF_TOKEN_REDACTED`
- SCOPE_LORA_DIR - LoRA models directory
  - Local: `~/.daydream-scope/models/lora`
  - RunPod: `/workspace/models/lora` (critical: NOT home directory on RunPod)
- CUDA_VISIBLE_DEVICES - GPU selection (e.g., `0` for first GPU)
- CONTAINER_IMAGE - Docker image path (`daydreamlive/scope:main`)
- CONTAINER_PORT - Web API port (8000)
- GRADIO_PORT - Gradio UI port (7860)
- LORA_MERGE_MODE - LoRA application strategy (`runtime_peft` or `permanent`)

**Build:**
- Docker build configuration for custom deployments
- Python wheel builds using hatchling (`build-system` in `pyproject.toml`)

## Platform Requirements

**Development:**
- macOS (Darwin) - Local development environment
- Python 3.10+
- GPU optional (for local testing)

**Production:**
- RunPod GPU cloud platform
- Recommended: RTX 4090 (24GB), RTX A6000 (48GB), or A100 (40GB/80GB)
- Container runtime: Docker with GPU support (`--gpus all`)
- Network: SSH gateway (`ssh.runpod.io`) or direct SSH access
- Storage: `/workspace/models/lora/` for LoRA files on RunPod

## Architecture Notes

**Plugin System:**
- Entry point registration via `pyproject.toml` `[project.entry-points."scope"]`
- Plugin pattern: `register_pipelines()` hook function
- Scope expects pipelines to be callable classes with `__call__()` method
- Optional `prepare()` method for video-input pipelines
- Optional `ui_field_config()` for UI parameter configuration

**Model Compatibility:**
- Krea Realtime: Wan2.1-T2V-14B (14 billion parameters)
- LoRA merge modes: Runtime (dynamic) or permanent (pre-merged)
- LoRA format: `.safetensors` (binary safe tensor serialization)

---

*Stack analysis: 2026-02-16*
