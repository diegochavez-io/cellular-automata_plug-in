"""
Cellular Automata Viewer - Entry Point

Usage:
    python -m cellular_automata [preset] [--size N] [--window WxH]

Examples:
    python -m cellular_automata
    python -m cellular_automata coral
    python -m cellular_automata jellyfish --size 512
    python -m cellular_automata reef --window 1200x1200

Engines:
    lenia       - Continuous CA with smooth kernels (default)
    life        - Game of Life and variants (B/S rules)
    excitable   - Greenberg-Hastings excitable media
    gray_scott  - Gray-Scott reaction-diffusion

Use --list to see all available presets.
"""

import sys
from .viewer import Viewer
from .presets import PRESET_ORDER, ENGINE_ORDER, list_presets


def main():
    preset = "coral"
    sim_size = 1024
    win_w, win_h = 900, 900

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--size" and i + 1 < len(args):
            sim_size = int(args[i + 1])
            i += 2
        elif arg == "--window" and i + 1 < len(args):
            parts = args[i + 1].split("x")
            win_w, win_h = int(parts[0]), int(parts[1])
            i += 2
        elif arg == "--list":
            print("\nAvailable presets:")
            for engine in ENGINE_ORDER:
                print(f"\n  [{engine}]")
                for key, name, desc in list_presets(engine):
                    print(f"    {key:16s} {name:20s} {desc}")
            print()
            return
        elif arg in ("--help", "-h"):
            print(__doc__)
            return
        elif arg in PRESET_ORDER:
            preset = arg
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            print(f"Use --list to see available presets")
            return

    print(f"Starting Cellular Automata Viewer")
    print(f"  Preset: {preset}")
    print(f"  Sim size: {sim_size}x{sim_size}")
    print(f"  Window: {win_w}x{win_h}")
    print()

    viewer = Viewer(
        width=win_w,
        height=win_h,
        sim_size=sim_size,
        start_preset=preset,
    )
    viewer.run()


if __name__ == "__main__":
    main()
