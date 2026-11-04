# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=None
STYLE={'pdf.fonttype': 42, 'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7, 'axes.labelsize': 8, 'axes.linewidth': 0.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.labelpad': 2}

def panel(height=1.65):
    # Each PDF is placed at 0.49 column width, close to its native size.
    fig, ax = plt.subplots(figsize=(1.68, height))
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.36 / height,
                        top=1 - 0.066 / height)
    ax.grid(linestyle=":", linewidth=0.45, alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    return fig, ax

def save(fig, name):
    # Keep the same canvas on paired panels so their axes align in LaTeX.
    fig.savefig(OUTPUT / f"{name}.pdf")
    plt.close(fig)

def hardware(archive):
    with archive.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_drone = defaultdict(list)
    for row in rows:
        by_drone[row["drone"]].append(row)

    fig, ax = panel()
    # Labels sit just outside each vehicle's distinct rectangular trajectory.
    offsets = {"cf1": (0.03, 0.12), "cf2": (-0.80, -0.24),
               "cf3": (0.03, 0.12), "cfb1": (0.03, -0.23),
               "cfb2": (-0.82, 0.12)}
    for index, (name, records) in enumerate(by_drone.items()):
        records.sort(key=lambda row: float(row["t"]))
        xy = np.array([[float(row["raw_x"]), float(row["raw_y"])]
                       for row in records])
        color = plt.get_cmap("tab10")(index)
        ax.plot(*xy.T, color=color, linewidth=0.65)
        ax.plot(*xy[0], "o", color=color, markersize=2)
        ax.plot(*xy[-1], "x", color=color, markersize=3, markeredgewidth=0.6)
        dx, dy = offsets[name]
        ax.text(*(xy[0] + [dx, dy]), name, color=color, fontsize=6.5)
    ax.set(xlabel="X [m]", ylabel="Y [m]", xlim=(-2.35, 0.85), ylim=(-1.9, 1.15))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-2, -1, 0])
    ax.set_yticks([-1, 0, 1])
    save(fig, "hardware_xy")

    unique = {}
    for row in rows:
        if row["phase"].startswith("mpc_"):
            unique.setdefault((row["phase"], float(row["t"])), row)
    control = sorted(unique.values(), key=lambda row: float(row["t"]))
    elapsed = np.array([float(row["t"]) for row in control])
    elapsed -= elapsed[0]
    fig, ax = panel()
    for key, label, color in [("loop_time_s", "Loop", "tab:orange"),
                               ("solve_time_s", "Solver", "tab:blue")]:
        times = [1000 * float(row[key]) for row in control]
        ax.plot(elapsed, times, label=label, color=color, linewidth=0.5)
    for limit, color, label in [(80, "tab:orange", "Solver limit (80 ms)"),
                                 (100, "tab:red", "Loop budget (100 ms)")]:
        ax.axhline(limit, color=color, linestyle="--", linewidth=0.6)
        ax.text(0.03, limit + 2, label, transform=ax.get_yaxis_transform(),
                color=color, fontsize=6, ha="left", va="bottom")
    ax.set(xlabel="Time [s]", ylabel="Time [ms]", xlim=(0, elapsed[-1]), ylim=(0, 112))
    ax.set_xticks([0, 20, 40])
    ax.set_yticks([0, 40, 80, 100])
    ax.legend(loc="center", bbox_to_anchor=(0.52, 0.55), frameon=False,
              handlelength=1.1, labelspacing=0.2, borderpad=0)
    save(fig, "hardware_timing")

def generate(output, csv_path):
    global OUTPUT
    OUTPUT=Path(output)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):
        hardware(Path(csv_path))
