# DayDream Scope - Krea Realtime Video Pipeline

Production deployment of Krea Realtime Video Pipeline on RunPod with custom 14B LoRAs.

## Quick Start

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 2. Setup MCP Server (Optional but Recommended)
See [MCP_SETUP.md](MCP_SETUP.md) for instructions on connecting Claude to DayDream documentation.

### 3. Deploy to RunPod
```bash
cd deploy
chmod +x runpod_deploy.sh
RUNPOD_HOST=root@your-ip ./runpod_deploy.sh
```

### 4. Sync Your Custom LoRAs
```bash
cd deploy
chmod +x sync_loras.sh
RUNPOD_HOST=root@your-ip ./sync_loras.sh
```

## Project Structure

```
daydream_scope/
├── CLAUDE.md              # Project context for Claude Code
├── MCP_SETUP.md          # MCP server setup instructions
├── README.md             # This file
├── .env.example          # Environment variables template
├── plugins/              # Custom Scope plugins
├── loras/               # Trained 14B LoRA models
├── docker/              # Custom container builds
└── deploy/              # Deployment scripts
    ├── runpod_deploy.sh     # Main deployment script
    └── sync_loras.sh        # LoRA sync script
```

## Documentation

- **CLAUDE.md**: Comprehensive project context and development guidelines
- **MCP_SETUP.md**: Real-time documentation access setup
- [DayDream LoRA Guide](https://docs.daydream.live/scope/guides/loras)
- [Plugin Development](https://docs.daydream.live/scope/guides/plugin-development)

## Current Status

- ✅ Project structure initialized
- ✅ CLAUDE.md documentation created
- ✅ Deployment scripts ready
- 🔄 Training 14B WAN LoRAs
- ⬜ MCP server connection pending
- ⬜ First RunPod deployment

## Key Features

- **14B Model Support**: Krea Realtime with Wan2.1-T2V-14B LoRAs
- **Runtime LoRA Switching**: Dynamic LoRA loading without restart
- **Multiple LoRA Support**: Combine multiple styles simultaneously
- **GPU-Optimized**: Deployed on RunPod with CUDA acceleration
- **Plugin System**: Extensible via custom Scope plugins

## License

Refer to DayDream Scope licensing terms.
