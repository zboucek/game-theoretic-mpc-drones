# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Render the first eight simulated seconds from saved cooperative PyBullet paths.

python -m tools.animate --source data/cooperative --output media/cooperative.gif
Uses Matplotlib/Pillow; FFmpeg and a new optimization run are unnecessary.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np


def render(source, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    source, output = Path(source), Path(output)
    trajectories = []
    hashes = {}
    for method in ("gne", "vgne"):
        file = source / method / "cooperative_terminal/trajectory.npz"
        hashes[f"{method}/cooperative_terminal/trajectory.npz"] = sha256(
            file.read_bytes()
        ).hexdigest()
        with np.load(file) as data:
            trajectories.append(
                (data["times"].copy(), data["states"].copy(), data["targets"].copy())
            )
        if trajectories[-1][0][-1] < 8:
            raise ValueError("Preview requires at least eight seconds")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), dpi=100, layout="constrained")
    paths = []
    drones = []
    for ax, method, (_, states, targets) in zip(axes, ("GNE", "vGNE"), trajectories):
        ax.set(
            xlim=(-7, 11), ylim=(-4, 4), xlabel="X [m]", ylabel="Y [m]", title=method
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        pp = []
        dd = []
        for agent, color in enumerate(("tab:blue", "tab:orange", "tab:green")):
            (line,) = ax.plot([], [], color=color, lw=1.5, alpha=0.65)
            pp.append(line)
            # An X with four rotors marks the moving vehicle's recorded position.
            (marker,) = ax.plot(
                [], [], marker="X", markersize=10, color=color, label=f"Drone {agent+1}"
            )
            dd.append(marker)
            ax.scatter(*targets[agent, :2], marker="*", s=85, color=color)
        ax.legend(fontsize=8, loc="upper left")
        paths.append(pp)
        drones.append(dd)
    frames = []
    for t in np.arange(80) / 10:
        for k, (times, states, _) in enumerate(trajectories):
            end = int(np.searchsorted(times, t, side="right"))
            for agent in range(3):
                pos = np.array(
                    [np.interp(t, times, states[:, agent, axis]) for axis in (0, 1)]
                )
                paths[k][agent].set_data(states[:end, agent, 0], states[:end, agent, 1])
                drones[k][agent].set_data([pos[0]], [pos[1]])
        fig.suptitle(f"PyBullet trajectories — simulated time {t:.1f} s", fontsize=12)
        fig.canvas.draw()
        frames.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        )
    plt.close(fig)
    # A single palette learned from all frames avoids frame-to-frame color flicker.
    samples = Image.new("RGB", (160 * 10, 80 * 8))
    for i, frame in enumerate(frames):
        samples.paste(frame.resize((160, 80)), ((i % 10) * 160, (i // 10) * 80))
    palette = samples.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    indexed = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        output,
        save_all=True,
        append_images=indexed[1:],
        duration=100,
        loop=0,
        optimize=False,
        disposal=2,
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                simulated_seconds=8,
                width=960,
                fps=10,
                frames=80,
                palette_colors=128,
                loop=0,
                source_sha256=hashes,
            ),
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/reference.zip"))
    parser.add_argument("--output", type=Path, default=Path("media/cooperative.gif"))
    args = parser.parse_args()
    if args.source.is_dir():
        print(render(args.source, args.output))
    else:
        import tempfile
        from tools.data_archives import extract

        with tempfile.TemporaryDirectory() as tmp:
            data = extract(args.source, Path(tmp) / "views")
            print(render(data / "cooperative", args.output))


if __name__ == "__main__":
    main()
