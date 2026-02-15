"""
Cellular Automata Viewer - Entry Point

Usage:
    python -m cellular_automata [preset] [--size N] [--window WxH]

Examples:
    python -m cellular_automata
    python -m cellular_automata orbium
    python -m cellular_automata cardiac --size 256
    python -m cellular_automata geminium --window 1200x1200

Presets:
    orbium      - Gliding organism (default)
    geminium    - Self-replicating pattern
    scutium     - Shield-shaped structures
    aquarium    - Rich ecosystem
    mitosis     - Blob division
    dual_ring   - Two-ring interference
    coral       - Branching growth
    cardiac     - Spiral waves
    primordial  - Self-organizing chaos
"""

import sys
from .viewer import Viewer
from .presets import PRESET_ORDER, list_presets


def main():
    preset = "orbium"
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
            for key, name, desc in list_presets():
                print(f"  {key:14s} {name:20s} {desc}")
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
            print(f"Available presets: {', '.join(PRESET_ORDER)}")
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
