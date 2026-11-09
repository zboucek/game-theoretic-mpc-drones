# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Recompute article results from numerical records; never modify reference data."""
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from experiments.equilibrium_selection.archive import serializable

    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def module(name, arguments, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "experiments." + name, *map(str, arguments)],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"{name} failed; see {log}\n" + Path(log).read_text()[-4000:]
        )


def selection(directory):
    """Recompute the V-A table, including distinct-profile refutation denominators."""
    from experiments.equilibrium_selection.protocol import PROTOCOL, METHODS
    from experiments.equilibrium_selection.reporting import audit

    records = {p.stem: read(p) for p in sorted((directory / "raw").glob("*.json"))}
    archive = SimpleNamespace(
        directory=directory,
        config=PROTOCOL,
        manifest={},
        get=lambda key: records[key],
        records=lambda kind=None: (
            r for r in records.values() if kind is None or r["kind"] == kind
        ),
    )
    checked = audit(archive)
    if not checked["passed"]:
        raise ValueError(f'Equilibrium record mismatch: {checked["failures"][:3]}')
    static = list(archive.records("static"))
    episodes = [
        read(p) for p in sorted((directory / "episodes/confirm").glob("*.json"))
    ]
    association = read(directory / "diagnostic_groups_confirm.json")
    expected = {r["record_id"] for r in static if r["validation"]["accepted"]}
    expected.update(
        e[endpoint]
        for e in episodes
        for endpoint in ("first_accepted", "last_accepted")
        if e[endpoint]
    )
    if not expected <= association.keys():
        raise ValueError("Missing unilateral-response association")
    # Several accepted calls may describe the same diagnosed control profile.
    representatives = {association[key] for key in expected}
    deviations = defaultdict(list)
    for record in archive.records("deviation"):
        deviations[record["candidate_id"]].append(record)
    methods = {}
    for method in METHODS:
        rows = [r for r in static if r["method"] == method]
        good = [r for r in rows if r["validation"]["accepted"]]
        loops = [e for e in episodes if e["method"] == method]
        diagnosed = [key for key in representatives if records[key]["method"] == method]
        if any(len(deviations[key]) != 21 for key in diagnosed):
            raise ValueError("Incomplete unilateral-response searches")
        methods[method] = dict(
            static=len(rows),
            accepted=len(good),
            episodes=len(loops),
            completed=sum(
                e["accepted_updates"] == e["requested_updates"] for e in loops
            ),
            diagnosed=len(diagnosed),
            refuted=sum(
                any(r["refutes_gne"] for r in deviations[key]) for key in diagnosed
            ),
            passage=sum(e["conflict_ever_passed"] for e in loops),
            median_accepted_s=float(
                np.median([r["solver"]["solve_seconds"] for r in good])
            ),
        )
    return dict(
        static_attempts=len(static),
        episodes=len(episodes),
        checked_records=checked["checked_attempts"],
        diagnosed=sum(r["diagnosed"] for r in methods.values()),
        refuted=sum(r["refuted"] for r in methods.values()),
        methods=methods,
    )


def cooperative(directory, logs):
    """Check all applied flight plans and measured arrivals/clearance (VI-A)."""
    for method in ("gne", "vgne"):
        module(
            "audit_blocked_terminal_mpc",
            ["--simulation", directory / method],
            logs / f"flight-{method}.log",
        )
    module(
        "audit_cooperative_equilibria",
        ["--source", directory],
        logs / "cooperative.log",
    )
    checks = read(directory / "equilibrium_validation.json")
    result = {}
    for row in checks["cases"]:
        method = row["method"]
        with np.load(
            directory / method / "cooperative_terminal/trajectory.npz"
        ) as data:
            t, states, targets = data["times"], data["states"], data["targets"]
        distance = np.linalg.norm(states[:, :, :3] - targets[None], axis=2)
        arrival = (distance <= 0.25) & (np.linalg.norm(states[:, :, 3:], axis=2) <= 0.3)
        if not arrival.any(axis=0).all():
            raise ValueError("A cooperative agent never arrives")
        result[method] = dict(
            applied=row["applied"],
            accepted=row["accepted_applied"],
            duration_s=float(t[-1]),
            minimum_distance_m=min(
                float(np.linalg.norm(states[:, i, :3] - states[:, j, :3], axis=1).min())
                for i in range(3)
                for j in range(i + 1, 3)
            ),
            arrivals_s=[float(t[np.flatnonzero(arrival[:, i])[0]]) for i in range(3)],
            maximum_final_error_m=float(distance[-1].max()),
        )
    return result


def adversarial(directory, logs, figures):
    """Check VI-B inputs and evaluate every panel at first simultaneous arrival."""
    module(
        "audit_blocked_terminal_mpc",
        ["--simulation", directory / "flight"],
        logs / "adversarial.log",
    )
    module(
        "plot_revision_adversarial",
        [
            "--source",
            directory / "flight",
            "--output",
            figures,
        ],
        logs / "adversarial-figure.log",
    )
    plotted = read(figures / "adversarial_arrival_manifest.json")
    return dict(
        joint_arrival_s=plotted["evaluation_window"]["end_s"],
        flight=plotted["metrics"],
    )


def analyze(data, output):
    """The caller supplies a disposable copy, so even legacy renderers write safely."""
    output.mkdir(parents=True, exist_ok=True)
    logs, figures = output / "logs", output / "figures"
    logs.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    result = dict(selection=selection(data / "equilibrium_selection"))
    print("Checked controlled equilibrium records", flush=True)
    result["cooperative"] = cooperative(data / "cooperative", logs)
    print("Checked cooperative flight plans", flush=True)
    result["adversarial"] = adversarial(data / "adversarial", logs, figures)
    from tools.response_relaxation import analyze as relaxation

    result["adversarial"]["relaxation"] = relaxation(data / "adversarial/relaxation", logs)
    print("Checked pursuit-evasion records", flush=True)
    module(
        "audit_current_scalability",
        ["--source", data / "scalability"],
        logs / "scalability.log",
    )
    module(
        "report_current_scalability",
        ["--source", data / "scalability", "--figures", figures],
        logs / "scalability-figure.log",
    )
    benchmark = read(data / "scalability/report_data.json")
    result["scalability"] = benchmark["totals"]
    result["scalability"]["h30_ratios"] = {
        row["method"]: row["qualified_runtime_ratio"]
        for row in benchmark["pairs"]
        if row["horizon"] == 30
    }
    from tools import hardware

    result["hardware"], _ = hardware.analyze(data / "hardware/run.csv")
    write(output / "metrics.json", result)
    return result


def tables(metrics, output):
    """Format the article table from saved metrics."""
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        "Method,Static accepted,Episodes completed,Conflict passed,Improvement found"
    ]
    tex = [
        r"{\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & \shortstack{Static\\accepted} & \shortstack{Episodes\\completed} & "
        r"\shortstack{Conflict\\passed} & \shortstack{Improvement\\found} \\",
        r"\midrule",
    ]
    selection = metrics["selection"]["methods"]
    rates = {
        key: [r[key] / r[denominator] for r in selection.values()]
        for key, denominator in (
            ("accepted", "static"),
            ("completed", "episodes"),
            ("passage", "episodes"),
            ("refuted", "diagnosed"),
        )
    }
    for name, label in (
        ("gne", "Joint KKT (GNE)"),
        ("vgne", "Joint KKT (vGNE)"),
        ("potential", "Potential minimization"),
    ):
        r = selection[name]
        values = [
            label,
            f"{r['accepted']}/{r['static']}",
            f"{r['completed']}/{r['episodes']}",
            f"{r['passage']}/{r['episodes']}",
            f"{r['refuted']}/{r['diagnosed']}",
        ]
        rows.append(",".join(values))
        for column, (key, denominator) in enumerate(
            (
                ("accepted", "static"),
                ("completed", "episodes"),
                ("passage", "episodes"),
                ("refuted", "diagnosed"),
            ),
            1,
        ):
            best = min(rates[key]) if key == "refuted" else max(rates[key])
            if r[key] / r[denominator] == best:
                values[column] = r"\textbf{" + values[column] + "}"
        tex.append(" & ".join(values) + r" \\")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "}",
            "% Static acceptance: all static calls. Completion/passage: all episodes.",
            "% Refutation: distinct diagnosed accepted profiles, including episode endpoints.",
        ]
    )
    (output / "equilibrium_selection.csv").write_text("\n".join(rows) + "\n")
    (output / "equilibrium_selection.tex").write_text("\n".join(tex) + "\n")
