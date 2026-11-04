# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
from game_theory_mpc import load_config, run_experiment, LinearNashMPCController
from game_theory_mpc.experiment_model import make_model, check_model

ROOT = Path(__file__).resolve().parents[1]


def test_six_complete_configs():
    configs = [load_config(p) for p in (ROOT / "configs").glob("*.toml")]
    assert len(configs) == 6
    assert len(next(c for c in configs if c.study == "scalability").cases) == 42
    assert len(next(c for c in configs if c.study == "controlled").cases) == 15


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, model=replace(c.model, dt=0.123)),
        lambda c: replace(c, execution=replace(c.execution, physics_hz=241)),
        lambda c: replace(c, model=replace(c.model, blocks=[1, 0])),
        lambda c: replace(c, model=replace(c.model, states=[[float("nan")] * 6] * 3)),
        lambda c: replace(c, model=replace(c.model, targets=[[0, 0, 1]])),
        lambda c: replace(c, execution=replace(c.execution, updates=0)),
        lambda c: replace(c, execution=replace(c.execution, physical_speed_bound=1.0)),
        lambda c: replace(c, methods=["adversarial_br"]),
    ],
)
def test_invalid_configs_rejected(change):
    with pytest.raises(ValueError):
        change(load_config(ROOT / "configs/cooperative.toml")).validate()


def test_controlled_structure_and_unknown_keys(tmp_path):
    c = load_config(ROOT / "configs/controlled.toml")
    with pytest.raises(ValueError, match="three"):
        replace(c, controlled=replace(c.controlled, agents=4)).validate()
    file = tmp_path / "config.toml"
    file.write_text('study="cooperative"\nunknown=2\n')
    with pytest.raises(ValueError, match="Invalid ExperimentConfig"):
        load_config(file)


@pytest.mark.parametrize("method", ["gne", "vgne", "potential"])
def test_changed_blocked_model_values_and_derivatives(method):
    c = load_config(ROOT / "configs/cooperative.toml")
    m = replace(
        c.model,
        dt=0.2,
        blocks=[1, 2],
        position_weight=2.3,
        velocity_weight=0.7,
        repulsion_weight=0.4,
        effort_weight=0.27,
        terminal_scale=0.6,
        input_bound=1.7,
        speed_bound=1.8,
        jerk_bound=4.3,
        separation=0.51,
        altitude_bounds=[0.25, 2.2],
    )
    model = make_model(m, method)
    model.commit_controls(np.full((3, 3), 0.1))
    u = np.linspace(-0.25, 0.25, model.nvar)
    check_model(model, u, m.states, m.targets, True)
    assert model.dt == 0.2 and model.horizon == 3 and model.blocking.blocks == 2
    assert model.input_bound == 1.7 and model.separation == 0.51
    baseline = make_model(replace(m, effort_weight=0.1), method)
    assert not np.allclose(
        model.evaluate_numpy(u, m.states, m.targets)["costs"],
        baseline.evaluate_numpy(u, m.states, m.targets)["costs"],
    )


def test_adversarial_cost_constraints_and_finite_difference_derivatives():
    c = load_config(ROOT / "configs/pursuit-evasion.toml")
    weights = dict(
        goal=12.0, evasion=3.0, pursuit=4.0, repulsion=0.7, obstacle=2.0, effort=0.4
    )
    m = replace(
        c.model,
        dt=0.2,
        blocks=[1, 2],
        adversarial_weights=weights,
        velocity_weight=0.4,
        terminal_scale=0.3,
        separation=0.51,
        input_bound=1.5,
        speed_bound=1.7,
        jerk_bound=4.0,
        altitude_bounds=[0.2, 2.3],
    )
    model = make_model(m, "adversarial_br")
    model.commit_controls(np.full((5, 3), 0.1))
    u = np.linspace(-0.2, 0.2, model.nvar)
    p = model.parameters(m.states, m.targets)
    check_model(model, u, m.states, m.targets)
    eps = 1e-5
    jac = []
    g = []
    for i in range(model.nvar):
        delta = np.zeros(model.nvar)
        delta[i] = eps
        a = model.evaluate_numpy(u + delta, m.states, m.targets)
        b = model.evaluate_numpy(u - delta, m.states, m.targets)
        jac.append((a["costs"] - b["costs"]) / (2 * eps))
        g.append((a["g"] - b["g"]) / (2 * eps))
    np.testing.assert_allclose(
        model.cost_jacobian(u, p), np.array(jac).T, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        model.constraint_jacobian(u, p), np.array(g).T, atol=1e-7
    )


def test_controlled_nondefault_parameters_match_numpy_and_jax():
    from experiments.equilibrium_selection.protocol import (
        Protocol,
        nominal,
        initial_controls,
    )
    from experiments.equilibrium_selection.solvers import Solvers

    p = Protocol(
        horizon=2,
        dt=0.2,
        bound=1.7,
        separation=0.45,
        effort_weight=0.23,
        repulsion_offset=0.07,
        max_evaluations=20,
    )
    solvers = Solvers(p)
    x, y = nominal("head_on")
    assert solvers.derivative_check(x, y)["passed"]
    # Compares the entire residual including configured box normals and prices.
    for method in ("gne", "vgne", "potential"):
        r = solvers.solve(method, x, y, initial_controls(x, "head_on", "braking", p))
        assert r["validation_error"] is None
        assert np.isfinite(r["controls"]).all()
        assert solvers.parameter_reuse_check(method, x, y)["passed"]
        controller = solvers.controllers[method]
        assert (
            controller.input_bound == p.bound
            and float(controller.R[0, 0]) == p.effort_weight
        )


def test_custom_run_needs_no_archive_and_preserves_existing_output(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    c = load_config(ROOT / "configs/cooperative.toml")
    m = replace(
        c.model, dt=0.2, blocks=[1, 2], targets=[[1, 0, 1], [-1, 0, 1], [0, -1, 1]]
    )
    c = replace(
        c,
        model=m,
        methods=["vgne"],
        execution=replace(c.execution, plant="nominal", updates=1),
    )
    output = run_experiment(c, tmp_path / "new")
    with np.load(output / "vgne/trajectory.npz") as data:
        np.testing.assert_allclose(data["times"], [0, 0.2])
        np.testing.assert_array_equal(data["targets"], m.targets)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    provenance = json.loads((output / "provenance.json").read_text())
    assert (
        provenance["seed"] == c.seed
        and provenance["code_sha256"]
        and provenance["dependencies"]["jax"]
    )
    before = (output / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(c, output)
    assert (output / "metrics.json").read_bytes() == before


def test_linear_changed_weights_horizon_and_dt():
    x = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    y = x[:, :3] + [0.2, 0.1, 0]
    a = LinearNashMPCController(
        2, horizon=3, dt=0.2, q_track=2.0, q_du=0.4, backend="fast"
    )
    b = LinearNashMPCController(
        2, horizon=3, dt=0.2, q_track=2.0, q_du=0.4, backend="nashopt"
    )
    for _ in range(2):
        u, p, _ = a.solve(x, y)
        v, q, _ = b.solve(x, y)
        np.testing.assert_allclose(u, v, atol=1e-6)
        np.testing.assert_allclose(p, q, atol=1e-6)
        x = np.asarray(p)[:, 0]


@pytest.mark.simulation
def test_pybullet_changed_timing_geometry_and_weights(tmp_path):
    pytest.importorskip("pybullet")
    pytest.importorskip("gym_pybullet_drones")
    c = load_config(ROOT / "configs/cooperative.toml")
    m = replace(
        c.model,
        dt=0.2,
        blocks=[1, 2],
        position_weight=1.3,
        repulsion_weight=0.4,
        effort_weight=0.2,
        targets=[[1, 0, 1], [-1, 0, 1], [0, -1, 1]],
        jerk_bound=4.0,
    )
    c = replace(
        c,
        model=m,
        methods=["vgne"],
        execution=replace(c.execution, updates=1, pid_hz=60, physics_hz=240),
    )
    output = run_experiment(c, tmp_path / "physics")
    directory = output / "vgne/cooperative_terminal"
    settings = json.loads((directory / "config.json").read_text())
    assert (
        settings["dt"] == 0.2
        and settings["pid_hz"] == 60
        and settings["physics_hz"] == 240
    )
    assert settings["cooperative_weights"]["position"] == 1.3
    with np.load(directory / "trajectory.npz") as data:
        assert len(data["times"]) == 13
        np.testing.assert_allclose(data["times"][-1], 0.2)
        np.testing.assert_array_equal(data["targets"], m.targets)
    attempt = json.loads((directory / "attempts/k000.json").read_text())
    assert attempt["independent"]["cost_max_error"] < 1e-8


def test_controlled_workflow_keeps_rejected_attempts_and_deviation_links(tmp_path):
    c = load_config(ROOT / "configs/controlled.toml")
    c = replace(
        c,
        cases=c.cases[:1],
        controlled=replace(
            c.controlled,
            horizon=2,
            updates=1,
            starts=["braking"],
            loop_starts=["braking"],
            max_evaluations=30,
        ),
    )
    output = run_experiment(c, tmp_path / "controlled")
    records = [json.loads(p.read_text()) for p in (output / "static").rglob("*.json")]
    assert {r["method"] for r in records} == {"gne", "vgne", "potential"}
    associations = json.loads((output / "diagnostic_groups.json").read_text())
    for p in (output / "deviations").rglob("*.json"):
        assert json.loads(p.read_text())["candidate_id"] in associations.values()
    assert json.loads((output / "metrics.json").read_text())["static"] == 3


def test_configured_weight_search_freezes_before_test(tmp_path):
    c = load_config(ROOT / "configs/weights.toml")
    x = [
        [-2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 1.0, 0.0, 0.0, 0.0],
    ]
    y = [r[:3] for r in x]
    cases = [
        dict(id="design", phase="train", states=x, targets=y),
        dict(id="heldout", phase="test", states=x, targets=y),
    ]
    c = replace(
        c,
        cases=cases,
        model=replace(c.model, blocks=[1, 1]),
        execution=replace(c.execution, updates=6),
        tuning=replace(
            c.tuning,
            baseline=[1.0, 1.0, 0.001],
            axes=[[1.0, 1.0000001], [1.0], [0.001]],
        ),
    )
    output = run_experiment(c, tmp_path / "tuning")
    frozen = json.loads((output / "frozen_weights.json").read_text())
    assert frozen["frozen_before_test"] and len(frozen["coordinate_trace"]) == 4
    assert len(frozen["design_scores"]) == 2
    result = json.loads((output / "metrics.json").read_text())
    assert result["manual"]["heldout"]["applied"] == 6
    assert result["automatic"]["heldout"]["success"]


def test_configured_scaling_restores_identical_starts_and_separates_warmup(tmp_path):
    c = load_config(ROOT / "configs/scalability.toml")
    # Keep one full and one blocked case, at the original ten-step horizon.
    cases = [
        next(
            row
            for row in c.cases
            if row["method"] == "vgne"
            and sum(row["blocks"]) == 10
            and row["parameterization"] == p
        )
        for p in ("full", "blocked")
    ]
    c = replace(c, cases=cases, benchmark=replace(c.benchmark, repeats=1))
    output = run_experiment(c, tmp_path / "benchmark")
    rows = json.loads((output / "metrics.json").read_text())
    assert len(rows) == 4 and sum(r["warmup"] for r in rows) == 2
    for directory in (output / "cases").iterdir():
        a, b = [
            json.loads(p.read_text())
            for p in sorted((directory / "samples").glob("*.json"))
        ]
        np.testing.assert_array_equal(
            a["initial_blocked_controls"], b["initial_blocked_controls"]
        )


def test_laboratory_offline_and_configured_linear_path(tmp_path):
    c = load_config(ROOT / "configs/laboratory.toml")
    c = replace(
        c,
        laboratory=replace(
            c.laboratory,
            csv=str(ROOT / "data/hardware/run.csv"),
            horizon=3,
            dt=0.2,
            q_track=2.0,
            q_du=0.4,
            updates=2,
        ),
    )
    output = run_experiment(c, tmp_path / "lab")
    with np.load(output / "linear-trajectory.npz") as data:
        np.testing.assert_allclose(data["times"], [0, 0.2, 0.4])
    result = json.loads((output / "metrics.json").read_text())
    assert result["offline"]["mpc_steps"] == 426
