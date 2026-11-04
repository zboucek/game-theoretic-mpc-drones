# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Reference verification and rendering, separate from custom experiment execution."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile


def writable_output(path):
    """Compatibility guard for portable auditors that write derived reports."""
    path = Path(path).resolve()
    reference_root = Path(__file__).resolve().parents[1] / "data"
    if (
        path.is_relative_to(reference_root)
        or path == Path(__file__).resolve().parents[1]
    ):
        raise ValueError(
            "Use a disposable copy or a new experiment output, not reference data"
        )
    return path


def output_directory(path):
    path = writable_output(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def verified_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "data/manifest.json").read_text())
    seen = set()
    for row in manifest["files"]:
        file = (root / row["path"]).resolve()
        if row["path"] in seen or not file.is_relative_to(root) or not file.is_file():
            raise ValueError("Invalid or missing reference: " + row["path"])
        seen.add(row["path"])
        if sha256(file.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Reference checksum mismatch: " + row["path"])
    return manifest


def compare(metrics, checks):
    import math

    for check in checks:
        value = metrics
        for key in check["key"].split("."):
            value = value[int(key)] if isinstance(value, list) else value[key]
        if not math.isfinite(value) or abs(value - check["expected"]) > check["atol"]:
            raise ValueError(
                f"Metric mismatch: {check['key']}: {value} (expected {check['expected']})"
            )


def reference(command, output, root=None, audit_data=None):
    from tools import results
    from tools.data_archives import extract

    root = Path(root or Path.cwd()).resolve()
    output = Path(output)
    manifest = verified_manifest(root)
    with tempfile.TemporaryDirectory(prefix="reference-", dir=output) as tmp:
        temporary = Path(tmp)
        if command == "verify":
            declaration = manifest["audit_dataset"]
            source = (
                Path(audit_data)
                if audit_data
                else next(
                    (
                        p
                        for p in (
                            root / declaration["filename"],
                            root.parent / declaration["filename"],
                        )
                        if p.is_file()
                    ),
                    None,
                )
            )
            if source is None or not source.exists():
                raise ValueError(
                    "Full numerical verification requires the separate audit dataset. "
                    "Use --audit-data /path/to/"
                    + declaration["filename"]
                    + ". Figures, tables and custom runs do not require it. "
                    "See zenodo/README.md for the prepared data deposit."
                )
            if source.is_dir():
                shutil.copytree(source / "data", temporary / "data")
            else:
                if sha256(source.read_bytes()).hexdigest() != declaration["sha256"]:
                    raise ValueError("Audit dataset archive checksum mismatch")
                extract(source, temporary)
            raw_manifest = temporary / "data/manifest.json"
            if (
                sha256(raw_manifest.read_bytes()).hexdigest()
                != declaration["manifest_sha256"]
            ):
                raise ValueError("Audit dataset manifest checksum mismatch")
            audit = verified_manifest(temporary)
            if audit["checks"] != manifest["checks"]:
                raise ValueError("Audit dataset and software reference versions differ")
            metrics = results.analyze(temporary / "data", output)
            compare(metrics, manifest["checks"])
            results.write(
                output / "verification.json",
                dict(
                    scope="complete independent numerical audit",
                    passed=True,
                    dataset_sha256=declaration["sha256"],
                    records=len(audit["files"]),
                ),
            )
        else:
            data = extract(root / "data/reference.zip", temporary / "views")
            metrics = json.loads((data / "metrics.json").read_text())
            compare(metrics, manifest["checks"])
            if command == "figures":
                from tools.reference_views import render

                render(data, root / "data/hardware/run.csv", output / "figures")
                expected = {
                    Path(n).name
                    for n in manifest["article_figures"]
                    + manifest["supporting_figures"]
                }
                actual = {p.name for p in (output / "figures").glob("*.pdf")}
                if expected != actual:
                    raise ValueError(f"Figure coverage mismatch: {expected ^ actual}")
            if command == "tables":
                results.tables(metrics, output / "tables")
            results.write(output / "metrics.json", metrics)
            results.write(
                output / "scope.json",
                dict(
                    scope="compact plotting inputs and archived summaries",
                    independent_solver_audit=False,
                    full_audit_command="python -m game_theory_mpc verify --audit-data DATASET.tar.xz",
                ),
            )
    verified_manifest(root)
    print(f"{command}: passed; outputs in {output}")
    return metrics


def quick(output, skip_pybullet=False, root=None):
    """Exercise all supported planner paths without checking any reference hashes."""
    from dataclasses import replace
    import numpy as np
    from game_theory_mpc import load_config, run_experiment
    from game_theory_mpc.config import ExperimentConfig, ControlledConfig
    from experiments.equilibrium_selection.protocol import nominal
    from experiments.equilibrium_selection.solvers import Solvers
    from experiments.equilibrium_selection.protocol import Protocol, initial_controls
    from game_theory_mpc import LinearNashMPCController

    root = Path(root or Path.cwd())
    states, targets = nominal("head_on")
    solvers = Solvers(Protocol(horizon=2, max_evaluations=30))
    for method in ("gne", "vgne", "potential"):
        result = solvers.solve(
            method,
            states,
            targets,
            initial_controls(states, "head_on", "braking", solvers.config),
        )
        if result["validation_error"] or not np.isfinite(result["controls"]).all():
            raise ValueError(method)
    for name in ("cooperative", "pursuit-evasion"):
        c = load_config(root / "configs" / f"{name}.toml")
        c = replace(
            c,
            execution=replace(
                c.execution, updates=2, plant="nominal" if skip_pybullet else "pybullet"
            ),
        )
        run_experiment(c, output / name)
    s = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    a = LinearNashMPCController(2, horizon=3, backend="fast").solve(
        s, s[:, :3] + [0.1, 0.2, 0]
    )[0]
    b = LinearNashMPCController(2, horizon=3, backend="nashopt").solve(
        s, s[:, :3] + [0.1, 0.2, 0]
    )[0]
    np.testing.assert_allclose(a, b, atol=1e-5)
    from tools.results import write

    write(output / "metrics.json", dict(passed=True, pybullet=not skip_pybullet))
