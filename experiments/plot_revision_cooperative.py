#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Render the cooperative-flight figure (VI-A) from logged states.

All-pair clearance and goal distances use every sample; paths alone are
downsampled for the vector plot. No controller is executed here."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-theory-mpc-manuscript-plots")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/game-theory-mpc-manuscript-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.manuscript_figure_layout import fit_3d_panel


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.reproduce import writable_output
    writable_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
    colors = ("#1768ac", "#d97706", "#24845b")
    fig = plt.figure(figsize=(3.5, 4.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.6, 1.))
    axes = (fig.add_subplot(grid[0, :], projection="3d"),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    inputs, minima = [], {}
    for method, style in (("gne", "-"), ("vgne", "--")):
        root = args.source/method
        directory = root/"cooperative_terminal"
        config_path, trajectory = directory/"config.json", directory/"trajectory.npz"
        inputs.extend([config_path, trajectory])
        config = json.loads(config_path.read_text())
        assert config["cooperative_layout"] == "all-moving"
        with np.load(trajectory) as archive:
            times, states, targets = archive["times"], archive["states"], archive["targets"]
        assert states.shape[1:] == (3, 6) and times[-1] == 30.
        np.testing.assert_array_equal(targets[2], [0., -2., 1.])
        positions = states[:, :, :3]
        for agent, color in enumerate(colors):
            # Downsample paths for the vector figure only; metrics use all samples.
            axes[0].plot(*positions[::6, agent].T, color=color, ls=style, lw=1.)
            axes[2].plot(times, np.linalg.norm(positions[:, agent]-targets[agent], axis=1),
                         color=color, ls=style, lw=1.)
            if method == "gne":
                axes[0].scatter(*positions[0, agent], color=color, s=12, marker="o")
                axes[0].scatter(*targets[agent], color=color, s=32, marker="*")
        measured = []
        for pair, color in zip(((0, 1), (0, 2), (1, 2)), colors):
            distance = np.linalg.norm(positions[:, pair[0]]-positions[:, pair[1]], axis=1)
            measured.append(float(distance.min()))
            axes[1].plot(times, distance, color=color, ls=style, lw=1.)
        minima[method] = measured
    axes[0].set(xlabel="$x$ [m]", ylabel="$y$ [m]", zlabel="$z$ [m]")
    axes[0].view_init(elev=24, azim=-64)
    axes[0].set_box_aspect((2., 1., .8), zoom=1.1)
    axes[0].set_anchor("C")
    axes[0].set_xticks([-5, 0, 5, 10])
    axes[0].set_yticks([-2, 0, 2])
    axes[0].set_zticks([.5, 1., 1.5])
    axes[0].tick_params(pad=0, labelsize=6)
    axes[0].xaxis.labelpad = axes[0].yaxis.labelpad = axes[0].zaxis.labelpad = -4
    agent_handles = [Line2D([], [], color=c) for c in colors]
    agent_labels = ["A1", "A2", "A3"]
    agent_legend = axes[0].legend(agent_handles, agent_labels, loc="upper center",
                                  bbox_to_anchor=(.5, 1.02), ncols=3, frameon=False,
                                  handlelength=1.2)
    agent_legend.set_in_layout(False)
    axes[1].set(xlim=(0, 2), ylim=(.45, 2.2), xlabel="Simulated time [s]",
                ylabel="Pair distance [m]", xticks=[0, 1, 2])
    axes[1].axhline(.5, color="black", lw=.8, ls="-.")
    axes[1].axhline(.55, color=".5", lw=.8, ls=":")
    axes[2].set(xlim=(0, 8), ylim=(.001, 12), yscale="log", xlabel="Simulated time [s]",
                ylabel="Goal distance [m]", xticks=[0, 4, 8])
    axes[2].axhline(.25, color=".4", lw=.8, ls=":")
    for axis in axes[1:]:
        axis.grid(alpha=.2)
    axes[1].legend([Line2D([], [], color=c) for c in colors], ["1–2", "1–3", "2–3"],
                   title="Pair", loc="center left", frameon=False, handlelength=1.2)
    axes[2].legend(agent_handles, agent_labels, loc="lower left", frameon=False, handlelength=1.2)
    captions = []
    for axis, caption in zip(axes, ("(a) Measured paths", "(b) Clearance", "(c) Goal convergence")):
        captions.append(axis.annotate(caption, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -34),
                                      textcoords="offset points", ha="center", va="top", fontsize=8,
                                      annotation_clip=False))
    zoom = fit_3d_panel(fig, axes[0], captions[0], box_aspect=(2., 1., .8), legend=agent_legend)
    outputs = []
    for extension in ("pdf", "png"):
        path = args.output/f"cooperative_three_moving.{extension}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    sources = (Path(__file__), ROOT/"experiments/manuscript_figure_layout.py")
    manifest = dict(command=[sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
                    source_sha256={os.path.relpath(p, ROOT): digest(p) for p in sources},
                    input_sha256={os.path.relpath(p, ROOT): digest(p) for p in inputs},
                    output_sha256={p.name: digest(p) for p in outputs},
                    layout="Single column: full-width 3D panel above side-by-side clearance and goal panels; captions below each panel.",
                    figure_inches=fig.get_size_inches().tolist(),
                    trajectory_view=dict(anchor="C", box_aspect=[2., 1., .8], zoom=zoom,
                                         fit="Full column width including axis labels"),
                    minimum_all_sample_pair_distances_m=minima,
                    display="Paths sampled every sixth logged state; separation/goal curves and metrics use all logged states. Separation axis shows first 2 s; goal axis first 8 s of 30 s.")
    (args.output/"cooperative_three_moving_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
