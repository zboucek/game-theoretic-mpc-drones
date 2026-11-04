# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Portable plotting of new trajectories; never substitutes reference artwork."""
from pathlib import Path
import numpy as np


def plot_run(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    for path in sorted(output.rglob("*trajectory.npz")):
        with np.load(path) as data:
            times, states, targets = data["times"], data["states"], data["targets"]
        fig, (xy, error) = plt.subplots(1, 2, figsize=(8, 3.4), layout="constrained")
        for i in range(states.shape[1]):
            xy.plot(states[:, i, 0], states[:, i, 1], label=f"A{i+1}")
            xy.scatter(states[0, i, 0], states[0, i, 1], marker="o", s=15)
        xy.scatter(
            targets[:, 0], targets[:, 1], marker="*", color="black", label="Targets"
        )
        pursuers = states.shape[1] - len(targets)
        errors = np.linalg.norm(states[:, pursuers:, :3] - targets[None], axis=2)
        error.plot(times, errors)
        xy.set(xlabel="X [m]", ylabel="Y [m]")
        xy.set_aspect("equal", adjustable="datalim")
        xy.legend(fontsize=7)
        error.set(xlabel="Simulated time [s]", ylabel="Target distance [m]")
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
