#!/usr/bin/env python3
import argparse
import os
import jax
import numpy as np
from discodj import DiscoDJ


def ascii_density_field(field, cmap="viridis", aspect=0.5):
    """
    Print a 2D field as colored ASCII art in the terminal.

    Parameters
    ----------
    field : 2D numpy array
        The field to visualize.
    cmap : str
        Matplotlib colormap name.
    aspect : float
        Vertical compression factor (<1 squashes vertically).
    """
    min_val, max_val = field.min(), field.max()
    normalized = (field - min_val) / (max_val - min_val)

    # Detect color mode
    colorterm = os.environ.get("COLORTERM", "").lower()
    truecolor = "truecolor" in colorterm or "24bit" in colorterm

    # Precompute color table
    try:
        from matplotlib import pyplot as plt
        cm = plt.get_cmap(cmap)
        color_table = (cm(np.linspace(0, 1, 256))[:, :3] * 255).astype(int)
    except ImportError:
        gray = np.linspace(0, 255, 256, dtype=int)
        color_table = np.stack([gray, gray, gray], axis=1)

    # Adjust vertical aspect (character cells are taller than wide)
    if aspect < 1:
        step = int(round(1 / aspect))
        normalized = normalized[::step, :]

    for row in normalized:
        for val in row:
            color_idx = int(val * 255)
            r, g, b = color_table[color_idx]
            if truecolor:
                # 24-bit color
                symbol = f"\033[38;2;{r};{g};{b}m██"
            else:
                # 256-color fallback
                r256, g256, b256 = r // 51, g // 51, b // 51
                color_code = 16 + 36 * r256 + 6 * g256 + b256
                symbol = f"\033[38;5;{color_code}m██"
            print(symbol, end="")
        print()
    print("\033[0m")  # Reset color


def main():
    parser = argparse.ArgumentParser(
        description="Run a 3D cosmological N-body simulation using DiscoDJ and visualize 2D density slices as ASCII art."
    )
    parser.add_argument(
        "--res", type=int, default=128,
        help="Simulation resolution (e.g. 64, 128, 256). Default: 128 (suited for GPU)"
    )
    parser.add_argument(
        "--boxsize", type=float, default=75.0,
        help="Box size in Mpc/h. Default: 75.0"
    )
    parser.add_argument(
        "--nsteps", type=int, default=5,
        help="Number of N-body integration steps. Default: 5"
    )
    parser.add_argument(
        "--cmap", type=str, default="viridis",
        help="Matplotlib colormap to use (e.g. viridis, plasma, magma, inferno). Default: viridis"
    )
    parser.add_argument(
        "--aspect", type=float, default=0.5,
        help="Vertical compression factor for terminal aspect ratio. Default: 0.5"
    )
    parser.add_argument(
        "--a-start", type=float, default=0.01,
        help="Starting scale factor. Default: 0.01"
    )
    parser.add_argument(
        "--a-end", type=float, default=1.0,
        help="Final scale factor. Default: 1.0"
    )

    args = parser.parse_args()
    disco_str = ("::::::::: ::::::::::: ::::::::   ::::::::   ::::::::                :::::::::  :::::::::::\n"
                 ":+:    :+:    :+:    :+:    :+: :+:    :+: :+:    :+:               :+:    :+:     :+:\n"
                 "+:+    +:+    +:+    +:+        +:+        +:+    +:+               +:+    +:+     +:+\n"
                 "+#+    +:+    +#+    +#++:++#++ +#+        +#+    +:+ +#++:++#++:++ +#+    +:+     +#+\n"
                 "+#+    +#+    +#+           +#+ +#+        +#+    +#+               +#+    +#+     +#+\n"
                 "#+#    #+#    #+#    #+#    #+# #+#    #+# #+#    #+#               #+#    #+# #+# #+#\n"
                 "######### ########### ########   ########   ########                #########   #####            ")
    print("\033[1;34mWelcome to\n", disco_str + "\n","\033[0m")
    print("This script runs a 3D cosmological N-body simulation using DiscoDJ and prints 2D slices of the density field.\n")
    print(f"🧩 Resolution: {args.res}³")
    print(f"📦 Box size:   {args.boxsize} Mpc/h")
    print(f"🎨 Colormap:   {args.cmap}")
    print(f"📈 Steps:      {args.nsteps}\n")
    print("Tip: For best visuals, zoom out in your terminal now (Ctrl + -)...\n")

    # --- Set up the simulation ---
    dj = DiscoDJ(dim=3, res=args.res, boxsize=args.boxsize)
    print("Running DiscoDJ on device:", jax.numpy.asarray(0).device.platform)
    dj = dj.with_timetables().with_linear_ps().with_ics(convert_to_numpy=True).with_lpt(n_order=1, convert_to_numpy=True)

    # --- Run the simulation ---
    a_list = np.linspace(args.a_start, args.a_end, args.nsteps, endpoint=False)
    for i, a in enumerate(a_list):
        X, P, _ = dj.run_nbody(
            a_ini=a,
            a_end=a_list[i + 1] if i + 1 < len(a_list) else args.a_end,
            n_steps=1,
            res_pm=dj.res,
            stepper="bullfrog",
        )
        dj = dj.with_external_ics(pos=X.reshape(-1, 3), vel=P.reshape(-1, 3))
        delta = dj.get_delta_from_pos(X, n_resample=4)
        print(f"\n\033[1;32mScale factor: a = {a:.3f}\033[0m")
        ascii_density_field(np.log10(1.1 + delta[delta.shape[0] // 2]), cmap=args.cmap, aspect=args.aspect)

    print("\n\033[1;34mSimulation complete!\033[0m\n")


if __name__ == "__main__":
    main()
