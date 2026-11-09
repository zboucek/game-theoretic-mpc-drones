# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Callable article experiment workflows; CLI and examples use this API."""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from .config import ExperimentConfig
from .experiment_io import write_json, provenance
from .experiment_model import make_model, check_model
from .control_step import control_step


def nominal(config, method, directory):
    directory.mkdir(parents=True)
    m, e = config.model, config.execution
    model = make_model(m, method)
    recovery = None
    states = np.asarray(m.states, float)
    targets = np.asarray(m.targets, float)
    model.prepare(states, targets)
    write_json(directory / "model.json", asdict(m))
    check_model(model, np.zeros(model.nvar), states, targets, derivatives=True)
    history = [states.copy()]
    times = [0.0]
    applied_controls = []
    accepted = 0
    attempted = 0
    termination = "update_limit"
    for k in range(e.updates):
        controls, prediction, record, recovery = control_step(
            model, recovery, m, e, method, states, targets
        )
        attempted += 1
        write_json(directory / "attempts" / f"k{k:03d}.json", record)
        if not record["applied"]:
            termination = "no_feasible_control"
            break
        states = states.copy()
        states[:, :3] += m.dt * states[:, 3:] + 0.5 * m.dt * m.dt * controls[:, 0]
        states[:, 3:] += m.dt * controls[:, 0]
        np.testing.assert_allclose(states, prediction[:, 0], atol=1e-10)
        model.commit_controls(controls[:, 0])
        history.append(states.copy())
        times.append((k + 1) * m.dt)
        applied_controls.append(controls[:, 0])
        accepted += record["equilibrium_accepted"]
    inputs = np.asarray(applied_controls).reshape(-1, len(states), 3)
    np.savez_compressed(
        directory / "trajectory.npz",
        times=times,
        states=history,
        controls=inputs,
        targets=targets,
    )
    ids = slice(m.pursuers, None) if m.pursuers else slice(None)
    errors = np.linalg.norm(np.asarray(history)[:, ids, :3] - targets[None], axis=2)
    result = dict(
        termination=termination,
        duration_s=times[-1],
        attempted=attempted,
        accepted=accepted,
        applied=len(inputs),
        final_goal_distances_m=errors[-1],
    )
    write_json(directory / "metrics.json", result)
    return result


def flight(config, method, directory):
    from experiments.experiment_blocked_terminal_mpc import run_case

    m, e = config.model, config.execution
    directory.mkdir(parents=True)
    args = SimpleNamespace(
        **asdict(e),
        **{
            k: v
            for k, v in asdict(m).items()
            if k not in ("states", "targets", "obstacles", "pursuers")
        },
        cooperative_layout="all-moving",
        equilibrium_method="potential" if method == "adversarial_br" else method,
        goal_weight=m.adversarial_weights["goal"],
        evasion_weight=m.adversarial_weights["evasion"],
    )
    from .blocked_game_mpc import AdversarialWeights

    args.adversarial_weights = AdversarialWeights(**m.adversarial_weights)
    # run_case shares the original physics loop and candidate-admission policy.
    case = dict(
        states=np.asarray(m.states, float),
        targets=np.asarray(m.targets, float),
        obstacles=np.asarray(m.obstacles, float).reshape(-1, 3),
        pursuers=m.pursuers,
        short_horizon=sum(m.blocks),
    )
    return run_case(
        "adversarial" if m.pursuers else "cooperative",
        "terminal",
        args,
        directory,
        case,
    )


def controlled(config, output):
    from experiments.equilibrium_selection.protocol import (
        Protocol,
        initial_controls,
        shift_controls,
    )
    from experiments.equilibrium_selection.solvers import (
        Solvers,
        deviation_jobs,
        deviation_attempt,
    )
    from experiments.equilibrium_selection.model import step

    c = config.controlled
    protocol = Protocol(
        **{k: v for k, v in asdict(c).items() if k in Protocol.__dataclass_fields__},
        seed=config.seed,
    )
    solvers = Solvers(protocol)
    counts = dict(
        static=0,
        static_accepted=0,
        loop_attempted=0,
        loop_accepted=0,
        applied=0,
        deviations=0,
    )
    associations = {}
    for case in config.cases:
        states, targets = np.asarray(case["states"]), np.asarray(case["targets"])
        checks = solvers.derivative_check(states, targets)
        if not checks["passed"]:
            raise ValueError("Controlled NumPy/JAX derivative mismatch")
        write_json(output / "checks" / f"{case['id']}.json", checks)
        diagnosed = []
        for method in config.methods:
            for start in c.starts:
                record = solvers.solve(
                    method,
                    states,
                    targets,
                    initial_controls(states, case["geometry"], start, protocol),
                )
                record["record_id"] = f"static_{case['id']}_{method}_{start}"
                write_json(
                    output / "static" / case["id"] / method / (start + ".json"), record
                )
                counts["static"] += 1
                counts["static_accepted"] += record["validation"]["accepted"]
                if record["validation"]["accepted"]:
                    diagnosed.append(record)
            for start in c.loop_starts:
                current = states.copy()
                guess = initial_controls(current, case["geometry"], start, protocol)
                history = [current.copy()]
                inputs = []
                times = [0.0]
                accepted_records = []
                dest = output / "episodes" / case["id"] / method / start
                for k in range(c.updates):
                    record = solvers.solve(method, current, targets, guess)
                    record[
                        "record_id"
                    ] = f"episode_{case['id']}_{method}_{start}_{k:03d}"
                    counts["loop_attempted"] += 1
                    record["applied"] = bool(record["validation"]["accepted"])
                    write_json(dest / "attempts" / f"k{k:03d}.json", record)
                    if not record["applied"]:
                        break
                    counts["loop_accepted"] += 1
                    accepted_records.append(record)
                    u = np.asarray(record["controls"]).reshape(3, protocol.horizon, 3)
                    current = step(current, u[:, 0], protocol)
                    history.append(current.copy())
                    inputs.append(u[:, 0])
                    times.append((k + 1) * protocol.dt)
                    counts["applied"] += 1
                    guess = shift_controls(u, protocol)
                dest.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    dest / "trajectory.npz",
                    times=times,
                    states=history,
                    targets=targets,
                    controls=np.asarray(inputs).reshape(-1, 3, 3),
                )
                if accepted_records:
                    diagnosed.extend((accepted_records[0], accepted_records[-1]))
        unique = []
        for record in diagnosed:
            representative = next(
                (
                    r
                    for r in unique
                    if record["method"] == r["method"]
                    and np.max(np.abs(np.asarray(record["controls"]) - r["controls"]))
                    <= c.unique_control_tolerance
                ),
                None,
            )
            if representative is not None:
                associations[record["record_id"]] = representative["record_id"]
                continue
            associations[record["record_id"]] = record["record_id"]
            unique.append(record)
            if c.deviations:
                for player, start, guess in deviation_jobs(
                    record, case["geometry"], protocol
                ):
                    deviation = deviation_attempt(
                        record, player, start, guess, protocol
                    )
                    deviation["candidate_id"] = record["record_id"]
                    write_json(
                        output
                        / "deviations"
                        / case["id"]
                        / f'{counts["deviations"]:06d}.json',
                        deviation,
                    )
                    counts["deviations"] += 1
    write_json(output / "diagnostic_groups.json", associations)
    write_json(output / "metrics.json", counts)
    return counts


def scalability(config, output):
    from experiments.benchmark_current_scalability import initial_controls

    results = []
    for case in config.cases:
        m = replace(
            config.model,
            states=case["states"],
            targets=case["targets"],
            pursuers=case.get("pursuers", 0),
            obstacles=case.get("obstacles", []),
            blocks=case["blocks"],
            velocity_weight=case.get("velocity_weight", config.model.velocity_weight),
        )
        method = case["method"]
        model = make_model(m, method)
        x, y = np.asarray(m.states), np.asarray(m.targets)
        seed = initial_controls(method, len(x), m.blocks)
        previous = np.zeros((len(x), 3))
        model.commit_controls(previous)
        check_model(model, seed, x, y, True)
        preparation = model.prepare(x, y)
        write_json(
            output / "cases" / case["id"] / "config.json",
            dict(model=asdict(m), method=method, preparation_seconds=preparation),
        )
        for k in range(config.benchmark.warmups + config.benchmark.repeats):
            model._warm = seed.copy()
            model._solved = True
            model.commit_controls(previous)
            if hasattr(model, "_repair_depth"):
                model._repair_depth = 0
            started = perf_counter()
            controls, prediction, info = model.solve(x, y)
            elapsed = perf_counter() - started
            checked = check_model(model, info["blocked_controls"], x, y)
            warmup = k < config.benchmark.warmups
            row = dict(
                case=case["id"],
                warmup=warmup,
                seconds=elapsed,
                feasible=info["feasible"],
                accepted=bool(info.get("validation", {}).get("accepted", False)),
            )
            write_json(
                output / "cases" / case["id"] / "samples" / f"k{k:03d}.json",
                dict(
                    info,
                    measurement=row,
                    controls=controls,
                    prediction=prediction,
                    states=x,
                    targets=y,
                    independent=checked,
                    applied=False,
                ),
            )
            results.append(row)
    write_json(output / "metrics.json", results)
    return results


def laboratory(config, output):
    from tools.hardware import analyze, plot
    from .linear_nash_mpc_controller import LinearNashMPCController

    l = config.laboratory
    metrics, rows = analyze(Path(l.csv))
    plot(rows, output / "hardware")
    model = LinearNashMPCController(
        len(l.states), l.horizon, l.dt, l.q_track, l.q_du, l.backend
    )
    states = np.asarray(l.states, float)
    targets = np.asarray(l.targets, float)
    history = [states.copy()]
    controls = []
    for k in range(l.updates):
        control, prediction, info = model.solve(states, targets)
        states = np.asarray(prediction)[:, 0]
        history.append(states.copy())
        controls.append(control[:, 0])
        write_json(output / "linear" / "attempts" / f"k{k:03d}.json", info)
    np.savez_compressed(
        output / "linear-trajectory.npz",
        times=np.arange(l.updates + 1) * l.dt,
        states=history,
        controls=controls,
        targets=targets,
    )
    write_json(
        output / "metrics.json",
        dict(offline=metrics, linear_final_positions_m=states[:, :3]),
    )
    return metrics


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """Run one validated configuration in a new directory and return its absolute path.

    A failed run retains provenance, diagnostics and an error record. Existing
    directories are never reused. Reference hashes and expected paper numbers
    play no role in a custom experiment.
    """
    config.validate()
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(config.seed)
    write_json(output / "effective_config.json", config.as_dict())
    write_json(output / "provenance.json", provenance(config))
    try:
        if config.study in ("cooperative", "pursuit-evasion"):
            function = flight if config.execution.plant == "pybullet" else nominal
            result = {
                method: function(config, method, output / method)
                for method in config.methods
            }
            write_json(output / "metrics.json", result)
        else:
            {
                "controlled": controlled,
                "scalability": scalability,
                "laboratory": laboratory,
            }[config.study](config, output)
        from .plotting import plot_run

        plot_run(output)
        write_json(output / "status.json", dict(status="complete"))
    except Exception as error:
        write_json(
            output / "status.json",
            dict(status="failed", error=type(error).__name__, message=str(error)),
        )
        raise
    return output
