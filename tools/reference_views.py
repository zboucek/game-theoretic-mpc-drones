# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Regenerate article figures/tables using compact numerical plotting inputs."""
import json
from pathlib import Path
import numpy as np
from . import results


def render(data, hardware_csv, output):
    from experiments.numerical_panels import plot_diagnostic
    from experiments.hardware_panels import generate as hardware
    from experiments.report_current_scalability import render_compact

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logs = output.parent / "logs"
    results.module(
        "plot_revision_cooperative",
        ["--source", data / "cooperative", "--output", output],
        logs / "cooperative-figure.log",
    )
    results.module(
        "plot_revision_adversarial",
        [
            "--source",
            data / "adversarial/flight",
            "--output",
            output,
        ],
        logs / "adversarial-figure.log",
    )
    with np.load(data / "conflict.npz") as archive:
        series = {
            m: tuple(archive[m + "_" + key] for key in ("states", "times", "progress"))
            for m in ("gne", "vgne", "potential")
        }
        targets = archive["targets"]
    plot_diagnostic(series, targets, output)
    scaling = json.loads((data / "scalability.json").read_text())
    render_compact(scaling["cases"], output)
    hardware(output, hardware_csv)
