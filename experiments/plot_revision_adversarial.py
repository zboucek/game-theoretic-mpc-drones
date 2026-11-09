#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Render pursuit-evasion results (VI-B) at first simultaneous arrival.

The same distance/speed stopping event truncates every panel and metric."""
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
from experiments.report_nominal_blocked_mpc import first_time
from experiments.manuscript_figure_layout import fit_3d_panel


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Selected original-PID flight root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.reproduce import writable_output
    writable_output(args.output)
    directory = args.source/"adversarial_terminal"
    inputs = [directory/name for name in ("config.json", "summary.json", "trajectory.npz")]
    config, summary = read(inputs[0]), read(inputs[1])
    assert config["pid_mode"] == "legacy" and config["horizon"] == 20
    assert config["adversarial_weights"]["goal"] == 100 and config["adversarial_weights"]["evasion"] == 10
    assert summary["applied_updates"] == config["updates"] and not config["hard_pair_separation"]
    with np.load(directory/"trajectory.npz") as archive:
        times, states, targets = archive["times"], archive["states"], archive["targets"]
        saturated, force_limited = archive["motor_saturation"], archive["force_limited"]
    assert times[-1] == summary["simulated_duration_s"] and states.shape == (len(times), 5, 6)
    positions = states[:, :, :3]
    goal = np.linalg.norm(positions[:, 3:]-targets[None], axis=2)
    speed = np.linalg.norm(states[:, 3:, 3:], axis=2)
    nearest = np.linalg.norm(positions[:, 3:, None]-positions[:, None, :3], axis=3).min(axis=2)
    np.testing.assert_allclose(goal[-1], summary["final_goal_distance_m"], atol=1e-12)
    np.testing.assert_allclose(speed[-1], summary["final_goal_speed_mps"], atol=1e-12)
    arrivals = []
    for agent in range(2):
        arrived = (goal[:, agent] <= .25) & (speed[:, agent] <= .3)
        arrivals.append(first_time(times, arrived))
    np.testing.assert_allclose(arrivals, summary["first_goal_stop_time_s"], atol=1e-12)
    at_goal = (goal <= .25) & (speed <= .3)
    joint = np.flatnonzero(at_goal.all(axis=1))
    assert len(joint), "The selected flight has no simultaneous arrival"
    end = int(joint[0])
    assert end > 0 and not at_goal[:end].all(axis=1).any()
    times, states, positions = times[:end+1], states[:end+1], positions[:end+1]
    goal, speed, nearest = goal[:end+1], speed[:end+1], nearest[:end+1]
    assert not (nearest < .3).any()
    duration = float(times[-1])
    metrics = dict(duration_s=duration, evaluated_samples=len(times), source_end_index=end,
                   goal_errors_m=goal[-1].tolist(), final_speeds_mps=speed[-1].tolist(), first_arrival_s=arrivals,
                   minimum_nearest_pursuer_m=nearest.min(axis=0).tolist(),
                   motor_saturation_fraction=float(saturated[1:end+1].mean()),
                   force_limited_fraction=float(force_limited[1:end+1].mean()),
                   minimum_altitude_m=float(states[:, :, 2].min()), maximum_altitude_m=float(states[:, :, 2].max()),
                   maximum_speed_mps=float(np.linalg.norm(states[:, :, 3:], axis=2).max()),
                   at_goal_before_terminal_sample=at_goal[end-1].tolist(),
                   termination="first_simultaneous_goal_arrival",
                   arrival_definition="Both evaders simultaneously have goal distance <= 0.25 m and speed <= 0.30 m/s")
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.5, 4.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.8, 1))
    axes = (fig.add_subplot(grid[0, :], projection="3d"), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    colors = ("#476b87", "#6a8eaa", "#9bacbc", "#c43d3d", "#d97706")
    for agent, color in enumerate(colors):
        style = "--" if agent >= 3 else "-"
        indices = np.unique(np.r_[np.arange(0, len(times), 6), len(times)-1])
        axes[0].plot(*positions[indices, agent].T, color=color, ls=style, lw=.8)
        axes[0].scatter(*positions[0, agent], s=15, color=color, marker="^" if agent < 3 else "s", depthshade=False)
        axes[0].scatter(*positions[-1, agent], s=15, color=color, marker="D", edgecolor="white", linewidth=.3, depthshade=False)
        label = f"P{agent+1}" if agent < 3 else f"E{agent-2}"
        axes[0].text(*(positions[0, agent]+[.1, .05, .09]), label, fontsize=6)
        # Direction markers use the same predefined simulated instant for all agents.
        k = int(np.searchsorted(times, 6.))
        axes[0].quiver(*positions[k-12, agent], *(positions[k+12, agent]-positions[k-12, agent]),
                       color=color, lw=.8, arrow_length_ratio=.5)
    obstacles = np.asarray(config["obstacles"])
    axes[0].scatter(*obstacles.T, color=".6", edgecolor=".3", linewidth=.3, s=18, depthshade=False)
    for agent, color in enumerate(colors[3:]):
        axes[0].scatter(*targets[agent], marker="*", s=50, color=color, edgecolor="black", linewidth=.3, depthshade=False)
        axes[0].text(*(targets[agent]+[.1, .05, .12]), f"T{agent+1}", fontsize=7)
        style = "--"
        axes[1].plot(times, nearest[:, agent], color=color, ls=style, label=f"E{agent+1}")
        axes[2].plot(times, goal[:, agent], color=color, ls=style)
        k = int(np.searchsorted(times, arrivals[agent]))
        axes[2].scatter(times[k], goal[k, agent], color=color, s=14, zorder=3)
    axes[0].set(xlabel="$x$ [m]", ylabel="$y$ [m]", zlabel="$z$ [m]")
    axes[0].view_init(elev=25, azim=-62)
    axes[0].set_box_aspect((2.1, 1.2, .9), zoom=1.1)
    axes[0].set_anchor("C")
    axes[0].set_xticks([-4, 0, 4])
    axes[0].set_yticks([-2, 0, 2])
    axes[0].set_zticks([.5, 1., 1.5])
    axes[0].tick_params(pad=0, labelsize=6)
    axes[0].xaxis.labelpad = axes[0].yaxis.labelpad = axes[0].zaxis.labelpad = -4
    role_legend = axes[0].legend([Line2D([], [], color=c, ls=style) for c, style in zip((colors[0], colors[3], colors[4]), ("-", "--", "--"))],
                   ["Pursuers P1–P3", "Evader E1", "Evader E2"], loc="upper center", bbox_to_anchor=(.5, 1.02), ncols=3, frameon=False, fontsize=6, handlelength=1.)
    axes[1].axhline(.3, color=".4", ls=":", lw=.8)
    axes[1].set(xlim=(0, duration), ylim=(0, nearest.max()*1.08), xlabel="Time [s]",
                ylabel="Nearest pursuer [m]")
    axes[1].legend(frameon=False, loc="upper right")
    axes[2].axhline(.25, color=".4", ls=":", lw=.8)
    axes[2].set(xlim=(0, duration), yscale="log", xlabel="Time [s]", ylabel="Goal distance [m]")
    captions = (f"(a) Measured 3D paths, {duration:.2f} s", "(b) 3D distance", "(c) Goal progress")
    caption_artists = []
    for axis, caption in zip(axes, captions):
        caption_artists.append(axis.annotate(caption, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -34),
                                             textcoords="offset points", ha="center", va="top", fontsize=8,
                                             annotation_clip=False))
    for axis in axes:
        axis.grid(alpha=.2)
    zoom = fit_3d_panel(fig, axes[0], caption_artists[0], box_aspect=(2.1, 1.2, .9), legend=role_legend)
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("pdf", "png"):
        path = args.output/f"adversarial_arrival.{extension}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    sources = [Path(__file__), ROOT/"experiments/report_nominal_blocked_mpc.py", ROOT/"experiments/manuscript_figure_layout.py"]
    manifest = dict(command=[sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
                    source_sha256={os.path.relpath(p, ROOT): digest(p) for p in sources},
                    input_sha256={os.path.relpath(p, ROOT): digest(p) for p in inputs},
                    output_sha256={p.name: digest(p) for p in outputs}, metrics=metrics,
                    evaluation_window=dict(start_s=0., end_s=duration, source_end_index=end,
                        stopping_rule="First logged state satisfying both evaders' distance/speed arrival criteria simultaneously"),
                    display="3D paths sampled every sixth evaluated state plus the terminal sample; all distance curves/metrics end at first simultaneous arrival. Vertical scale expanded for visibility. Arrows at 6 s. No new simulation.",
                    trajectory_view=dict(elevation_degrees=25, azimuth_degrees=-62, box_aspect=[2.1, 1.2, .9], zoom=zoom,
                                         fit="Full column width including axis labels"),
                    figure_layout=dict(size_inches=[3.5, 4.5], height_ratios=[1.8, 1], trajectory_anchor="C",
                                       caption_position="below each panel"),
                    role_styles=dict(pursuers="solid", evaders="dashed"))
    (args.output/"adversarial_arrival_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(flight=metrics)))


if __name__ == "__main__":
    main()
