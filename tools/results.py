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
    """Recompute the IV-A table, including distinct-profile refutation denominators."""
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
    """Check all applied flight plans and measured arrivals/clearance (V-A)."""
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
    """Check V-E inputs and evaluate every panel at first simultaneous arrival."""
    module(
        "audit_blocked_terminal_mpc",
        ["--simulation", directory / "flight"],
        logs / "adversarial.log",
    )
    for profile in ("original", "goal20", "goal100", "goal100_escape10"):
        module(
            "audit_blocked_terminal_mpc",
            ["--benchmark", directory / "weights" / profile],
            logs / f"weights-{profile}.log",
        )
    module(
        "plot_revision_adversarial",
        [
            "--source",
            directory / "flight",
            "--weights",
            directory / "weights",
            "--output",
            figures,
        ],
        logs / "adversarial-figure.log",
    )
    plotted = read(figures / "adversarial_arrival_manifest.json")
    return dict(
        joint_arrival_s=plotted["evaluation_window"]["end_s"],
        flight=plotted["metrics"],
        nominal_weights=plotted["nominal_weight_comparison"],
    )


def weighted_episodes(base, inspector, spec):
    """Check every design/evaluation candidate and recompute trajectory summaries."""
    from experiments import cooperative_weight_study as study

    checked = 0
    for path in sorted(base.rglob("trajectory.npz")):
        inner = path.parent
        if not (inner / "config.json").exists():
            continue
        physics = inner.name == "cooperative_terminal"
        directory = inner.parent if physics else inner
        # The flight's own summary has a different schema; the outer one has rollout metrics.
        if not (directory / "summary.json").exists():
            raise ValueError(f"Missing episode summary: {directory}")
        config = read(inner / "config.json")
        weights = (
            [
                config["stage_weights"]["position"],
                config["velocity_weight"],
                config["cooperative_weights"]["repulsion"],
            ]
            if physics
            else config["weights"]
        )
        effort = (
            config["stage_weights"]["input"]
            if physics
            else config.get("effort_weight", 0.1)
        )
        summary = read(directory / "summary.json")
        paths = sorted(
            (inner / "attempts").glob("k*.json" if physics else "k*.json.gz")
        )
        records = [read(p) if physics else study.read_attempt(p) for p in paths]
        accepted = applied = 0
        with np.load(path) as data:
            for i, record in enumerate(records):
                # Physics logs 12 PID samples per 0.1 s planning interval.
                np.testing.assert_allclose(
                    record["states"],
                    data["states"][i * 12 if physics else applied],
                    atol=2e-9,
                )
                previous = (
                    np.zeros((3, 3))
                    if i == 0
                    else np.asarray(records[i - 1]["controls"])[:, 0]
                )
                np.testing.assert_allclose(
                    record["previous_controls"], previous, atol=2e-9
                )
                if i and not records[i - 1]["applied"]:
                    raise ValueError("A stopped episode executes another input")
                if "tightened_attempt" in record:
                    inspector.tightened(record, weights, effort_weight=effort)
                recovery = (
                    record["constraint_jerk_bound"] == 20.0
                    if physics
                    else record["recovery"]
                )
                ok = inspector.candidate(
                    record, weights, recovery, effort_weight=effort
                )
                extent = study.first_interval_extrema(
                    record["states"], np.asarray(record["controls"])[:, 0]
                )
                for key, value in extent.items():
                    np.testing.assert_allclose(
                        value, record["first_interval"][key], atol=2e-9
                    )
                speed = max(
                    np.linalg.norm(np.asarray(record["states"])[:, 3:], axis=1).max(),
                    np.linalg.norm(
                        np.asarray(record["prediction"])[:, 0, 3:], axis=1
                    ).max(),
                )
                admitted = bool(
                    record["feasible"]
                    and extent["first_interval_minimum_distance_m"] >= 0.5 - 1e-8
                    and extent["first_interval_minimum_altitude_m"] >= 0.3 - 1e-8
                    and extent["first_interval_maximum_altitude_m"] <= 2.0 + 1e-8
                    and speed <= 2.5 + 1e-8
                )
                if admitted != record["applied"]:
                    raise ValueError(
                        "Input execution decision disagrees with constraints"
                    )
                if admitted:
                    applied += 1
                    accepted += ok
            controls = data["accelerations"][1:] if physics else data["controls"]
            calculated = study.metrics(
                data["times"],
                data["states"],
                controls,
                data["targets"],
                spec["updates"] * 0.1,
                summary["termination"],
                accepted,
                applied,
            )
            for key, value in calculated.items():
                if value is None or isinstance(value, (str, bool)):
                    if summary[key] != value:
                        raise ValueError(f"Weight episode metric mismatch: {key}")
                else:
                    np.testing.assert_allclose(summary[key], value, atol=2e-9)
            if not physics:
                np.testing.assert_allclose(
                    data["states"][1:, :, :3],
                    data["states"][:-1, :, :3]
                    + 0.1 * data["states"][:-1, :, 3:]
                    + 0.005 * controls,
                    atol=2e-9,
                )
                np.testing.assert_allclose(
                    data["states"][1:, :, 3:],
                    data["states"][:-1, :, 3:] + 0.1 * controls,
                    atol=2e-9,
                )
            if calculated["completed"]:
                assert (
                    data["states"][:, :, 2].min() >= 0.3 - 1e-8
                    and data["states"][:, :, 2].max() <= 2.0 + 1e-8
                )
                assert (
                    np.linalg.norm(data["states"][:, :, 3:], axis=2).max() <= 2.5 + 1e-8
                )
        checked += 1
    return checked


def weights(directory):
    """Audit all coordinate-search candidates and all twelve manual/tuned pairs."""
    from experiments import cooperative_weight_study as study
    from experiments.audit_cooperative_weight_study import Audit

    calibration = directory / "calibration"
    spec = read(calibration / "manifest.json")["protocol"]
    inspector = Audit()
    episodes = weighted_episodes(calibration, inspector, spec)
    frozen = read(calibration / "frozen_weights.json")
    scores = {}
    for key, row in frozen["design_scores"].items():
        records = [
            read(calibration / "design" / key / case["id"] / "summary.json")
            for case in spec["cases"]["train"]
        ]
        eligible = all(r["success"] for r in records)
        assert row["eligible"] == eligible
        scores[key] = (
            float(np.mean([r["score"] for r in records])) if eligible else float("inf")
        )
        if eligible:
            np.testing.assert_allclose(scores[key], row["score"], atol=1e-12)
    current = tuple(study.BASELINE)
    trace = [study.key(current)]
    for axis, values in enumerate(study.AXES):
        options = [current]
        for value in values:
            point = list(current)
            point[axis] = value
            options.append(tuple(point))
        current = min(options, key=lambda v: scores[study.key(v)])
        trace.append(study.key(current))
    np.testing.assert_array_equal(current, frozen["full"])
    assert trace == frozen["coordinate_trace"]
    result = dict(
        checked_episodes=episodes,
        checked_candidates=inspector.records,
        design_vectors=len(scores),
        selected_weights=list(current),
        design_improvement_percent=100
        * (1 - scores[study.key(current)] / scores[study.key(study.BASELINE)]),
    )
    for phase in ("test", "physics"):
        ids = (
            [c["id"] for c in spec["cases"]["test"]]
            if phase == "test"
            else spec["cases"]["physics_ids"]
        )
        result[phase] = {}
        rows = {}
        for name, folder in (("manual", "baseline"), ("automatic", "full")):
            rows[name] = [
                read(calibration / phase / folder / case / "summary.json")
                for case in ids
            ]
            r = rows[name]
            result[phase][name] = dict(
                cases=len(r),
                success=sum(v["success"] for v in r),
                accepted=sum(v["accepted"] for v in r),
                applied=sum(v["applied"] for v in r),
            )
            for label, field in (
                ("arrival_s", "joint_arrival_s"),
                ("ise_m2s", "integrated_position_error_m2s"),
                ("effort", "input_effort_m2_s3"),
            ):
                result[phase][name][label] = float(np.median([v[field] for v in r]))
        if not all(v["success"] for r in rows.values() for v in r):
            raise ValueError("Manual/automatic comparison needs every declared case")
        pairs = []
        for key, a, b in zip(ids, rows["manual"], rows["automatic"]):
            pairs.append(
                dict(
                    case=key,
                    error_change_percent=100
                    * (
                        b["integrated_position_error_m2s"]
                        / a["integrated_position_error_m2s"]
                        - 1
                    ),
                    effort_change_percent=100
                    * (b["input_effort_m2_s3"] / a["input_effort_m2_s3"] - 1),
                    arrival_change_s=b["joint_arrival_s"] - a["joint_arrival_s"],
                )
            )
        result[phase]["pairs"] = pairs
        result[phase]["paired_medians"] = {
            k: float(np.median([r[k] for r in pairs]))
            for k in (
                "error_change_percent",
                "effort_change_percent",
                "arrival_change_s",
            )
        }
    return result


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
    print("Checked pursuit-evasion records", flush=True)
    result["weights"] = weights(data / "weights")
    print("Checked calibration and held-out weight episodes", flush=True)
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
    """Format the article table and supporting weight table from saved metrics."""
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
