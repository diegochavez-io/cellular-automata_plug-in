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


def snap(preset, sim_size, steps):
    """Headless mode: run N steps, save screenshot, exit."""
    import os
    import numpy as np

    # Import engines and pipeline directly (no pygame window)
    from .presets import get_preset, UNIFIED_ORDER
    from .iridescent import IridescentPipeline

    try:
        from scipy.ndimage import gaussian_filter as _scipy_gaussian
        from scipy.ndimage import zoom as _scipy_zoom
    except ImportError:
        _scipy_gaussian = None
        _scipy_zoom = None

    BASE_RES = 512
    res_scale = sim_size / BASE_RES

    screenshots_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "screenshots"
    )
    os.makedirs(screenshots_dir, exist_ok=True)

    presets_to_snap = [preset] if preset != "all" else UNIFIED_ORDER

    for pkey in presets_to_snap:
        p = get_preset(pkey)
        if not p:
            print(f"Unknown preset: {pkey}")
            continue

        engine_name = p["engine"]

        # Create engine
        from .viewer import ENGINE_CLASSES
        cls = ENGINE_CLASSES[engine_name]

        if engine_name == "lenia":
            eng = cls(size=sim_size, R=max(5, int(p.get("R", 13) * res_scale)),
                      T=p.get("T", 10), mu=p.get("mu", 0.15), sigma=p.get("sigma", 0.017),
                      kernel_peaks=p.get("kernel_peaks"), kernel_widths=p.get("kernel_widths"))
        elif engine_name == "smoothlife":
            ri = max(3, int(p.get("ri", 8) * res_scale))
            ra = max(ri + 3, int(p.get("ra", 24) * res_scale))
            eng = cls(size=sim_size, ri=ri, ra=ra,
                      b1=p.get("b1", 0.278), b2=p.get("b2", 0.365),
                      d1=p.get("d1", 0.267), d2=p.get("d2", 0.445),
                      alpha_n=p.get("alpha_n", 0.028), alpha_m=p.get("alpha_m", 0.147),
                      dt=p.get("dt", 0.1))
        elif engine_name == "mnca":
            raw_rings = p.get("rings") or [(0, 5), (8, 15)]
            scaled_rings = [(max(0, int(ir * res_scale)), max(2, int(orr * res_scale)))
                            for ir, orr in raw_rings]
            eng = cls(size=sim_size, rings=scaled_rings, rules=p.get("rules"),
                      delta=p.get("delta", 0.05))
        else:
            eng = cls(size=sim_size)

        # Seed
        seed_kwargs = {}
        if "density" in p:
            seed_kwargs["density"] = p["density"]
        eng.seed(p.get("seed", "random"), **seed_kwargs)

        # Containment mask (matches viewer)
        center = sim_size / 2.0
        Y, X = np.ogrid[:sim_size, :sim_size]
        dist = np.sqrt((X - center) ** 2 + (Y - center) ** 2) / center
        if engine_name == "mnca":
            # Wide gaussian — soft fade, no hard edge
            mask = np.exp(-0.5 * (dist / 0.50) ** 2)
            mask[dist > 0.75] *= np.clip(1.0 - (dist[dist > 0.75] - 0.75) / 0.15, 0.0, 1.0)
        else:
            fade = np.clip((dist - 0.25) / 0.25, 0.0, 1.0)
            mask = 1.0 - fade * 0.06

        # Noise mask for center perturbation
        dist_sq = ((X - center) ** 2 + (Y - center) ** 2) / (center * center)
        noise_mask = 0.008 * np.exp(-dist_sq / (2 * 0.18 ** 2))

        # Run simulation with containment + churn (matches viewer)
        print(f"  {pkey}: running {steps} steps...", end="", flush=True)
        for step_i in range(steps):
            eng.step()
            eng.world *= mask
            # Center noise
            noise = np.random.randn(sim_size, sim_size) * noise_mask
            eng.world = np.clip(eng.world + noise, 0.0, 1.0)
            # Periodic churn (add/erase blobs)
            if engine_name in ("lenia", "smoothlife") and np.random.random() < 0.03:
                cx = sim_size // 2 + np.random.randint(-sim_size // 5, sim_size // 5)
                cy = sim_size // 2 + np.random.randint(-sim_size // 5, sim_size // 5)
                r = max(8, sim_size // 25)
                if np.random.random() < 0.5:
                    eng.add_blob(cx, cy, radius=r, value=0.4)
                else:
                    eng.remove_blob(cx, cy, radius=r)
            elif engine_name == "mnca" and np.random.random() < 0.02:
                cx = sim_size // 2 + np.random.randint(-sim_size // 6, sim_size // 6)
                cy = sim_size // 2 + np.random.randint(-sim_size // 6, sim_size // 6)
                eng.add_blob(cx, cy, radius=max(8, sim_size // 30), value=0.4)
            # Auto-reseed if dead
            if step_i % 50 == 49 and float(eng.world.mean()) < 0.002:
                seed_kw = {}
                if "density" in p:
                    seed_kw["density"] = p["density"]
                eng.seed(p.get("seed", "random"), **seed_kw)

        # Build spatial color offset (matches viewer — noise-based)
        from .viewer import Viewer
        # Use Viewer's static method for consistent color offset
        radial_c = np.clip(dist, 0, 1) * 0.25
        noise_size = max(8, sim_size // 64)
        noise_small = np.random.randn(noise_size, noise_size).astype(np.float32)
        if _scipy_gaussian:
            noise_small = _scipy_gaussian(noise_small, 1.5)
        if _scipy_zoom:
            noise = _scipy_zoom(noise_small, sim_size / noise_size, order=1)[:sim_size, :sim_size]
        else:
            fac = sim_size // noise_size
            noise = np.repeat(np.repeat(noise_small, fac, axis=0), fac, axis=1)[:sim_size, :sim_size]
        noise_range = max(noise.max() - noise.min(), 0.001)
        noise = (noise - noise.min()) / noise_range * 0.6 - 0.3
        color_offset = (radial_c + noise).astype(np.float32)

        # Render with iridescent pipeline (no forced blur — preserves detail)
        iri = IridescentPipeline(sim_size)
        iri.set_palette(p.get("palette", "oil_slick"))

        rgb = iri.render(eng.world, 0.016, t_offset=color_offset)

        # Gentle bloom
        factor = 8
        h, w = rgb.shape[:2]
        small = rgb[::factor, ::factor, :].astype(np.float32)
        if _scipy_gaussian:
            glow = _scipy_gaussian(small, [3.5, 3.5, 0])
        else:
            glow = small
        glow = np.repeat(np.repeat(glow, factor, axis=0), factor, axis=1)[:h, :w, :]
        result = rgb.astype(np.float32) + glow * 0.4
        np.clip(result, 0, 255, out=result)
        rgb = result.astype(np.uint8)

        # Save as PNG (no pygame needed)
        try:
            from PIL import Image
            img = Image.fromarray(rgb)
            path = os.path.join(screenshots_dir, f"ca_{pkey}.png")
            img.save(path)
            latest = os.path.join(screenshots_dir, "latest.png")
            img.save(latest)
            print(f" saved: {path}")
        except ImportError:
            # Fallback: save raw with pygame
            import pygame
            pygame.init()
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1).copy())
            path = os.path.join(screenshots_dir, f"ca_{pkey}.png")
            pygame.image.save(surface, path)
            latest = os.path.join(screenshots_dir, "latest.png")
            pygame.image.save(surface, latest)
            pygame.quit()
            print(f" saved: {path}")


def main():
    preset = "coral"
    sim_size = 1024
    win_w, win_h = 900, 900
    snap_steps = 0

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
        elif arg == "--snap" and i + 1 < len(args):
            snap_steps = int(args[i + 1])
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
        elif arg in PRESET_ORDER or arg == "all":
            preset = arg
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            print(f"Use --list to see available presets")
            return

    if snap_steps > 0:
        print(f"Headless snap mode: {preset} @ {sim_size}x{sim_size}, {snap_steps} steps")
        snap(preset, sim_size, snap_steps)
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
