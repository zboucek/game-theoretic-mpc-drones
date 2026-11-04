#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Rollout weight calibration and transfer: article IV-B.

cases/protocol define design and held-out episodes; train searches only design
cases. rollout and physics share the cooperative vGNE controller and metrics.
Manual and automatic transfer are retained in this article edition."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cooperative-weight-study-mpl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.equilibrium_selection.archive import archive_lock, digest, serializable, write_json
from experiments.equilibrium_mpc_execution import first_interval_extrema

BLOCKS = (1, 1, 2, 4, 6, 6)
BASELINE = (1., 1., 1.)  # position, velocity, repulsion; effort remains 0.1
AXES = ((.3, 1., 3., 10.), (.3, 1., 3.), (.1, 1., 3.))
SOURCES = ("experiments/cooperative_weight_study.py", "src/game_theory_mpc/blocked_game_mpc.py",
    "src/game_theory_mpc/cooperative_equilibrium_mpc.py", "experiments/equilibrium_mpc_execution.py",
    "experiments/experiment_blocked_terminal_mpc.py", "src/game_theory_mpc/acceleration_pid.py",
    "experiments/equilibrium_selection/archive.py")


def read(path):
    return json.loads(Path(path).read_text())


def key(weights):
    return "_".join(f"{v:g}".replace(".", "p") for v in weights)


def cases():
    """Freeze geometries before training; test variation is not training jitter."""
    def head(name, gap, speed, third_speed=0.):
        return dict(id=name, family="head-on", states=[[-gap/2, 0, 1, speed, 0, 0],
            [gap/2, 0, 1, -speed/2, 0, 0], [0, 2, 1, 0, -third_speed, 0]],
            targets=[[10, 0, 1], [-6, 0, 1], [0, -2, 1]])
    def crossing(name, radius, speed):
        return dict(id=name, family="crossing", states=[[-radius, 0, 1, speed, 0, 0],
            [0, -radius, 1, 0, speed*.8, 0], [radius, radius, 1, -speed*.5, -speed*.5, 0]],
            targets=[[3, 0, 1], [0, 3, 1], [-2, -2, 1]])
    train = [head("design_head", .7, .15), crossing("design_cross", 1., .15),
             crossing("design_cross_wide", 1.4, .3)]
    test = [head(f"test_head_{i}", gap, speed, third) for i, (gap, speed, third) in enumerate(
        ((.6, .2, 0.), (.65, .1, .2), (.8, .3, .1), (1., .2, .3), (1.2, .4, .15), (1.5, .3, .25)))]
    test += [crossing(f"test_cross_{i}", radius, speed) for i, (radius, speed) in enumerate(
        ((.6, .1), (.75, .25), (.9, .35), (1.1, .2), (1.3, .1), (1.6, .4)))]
    return dict(train=train, test=test, physics_ids=["test_head_0", "test_cross_1"])


def protocol(updates):
    return dict(version=1, dt=.1, updates=updates, blocks=BLOCKS, baseline=BASELINE,
        coordinate_values=AXES, coordinate_passes=1, cases=cases(), method="vgne",
        effort_weight=.1, terminal_rule="DARE(position, velocity, effort)-Q; recomputed for each weight vector",
        bounds=dict(acceleration=2., speed=2., jerk=6., altitude=[.35, 1.95], separation=.55),
        recovery=dict(speed=2.2, jerk=20., altitude=[.3, 2.], separation=.55),
        physical=dict(separation=.5, altitude=[.3, 2.], speed=2.5),
        acceptance=dict(inequality=1e-7, kkt=1e-5, minimum_multiplier=-1e-7),
        arrival=dict(distance=.25, speed=.3, dwell_s=.5),
        objective="ISE/(T*sum(initial_distance^2)) + 0.05*effort/(T*N*3*2^2)",
        selection="Any failed or non-arriving design episode makes a weight vector ineligible; baseline wins ties",
        optimizer="One deterministic coordinate pass; tracking-only uses the same position candidates; no test feedback")


def initialize(output, updates):
    spec = protocol(updates)
    hashes = {name: digest(ROOT/name) for name in SOURCES}
    path = output/"manifest.json"
    if path.exists():
        old = read(path)
        if old["protocol"] != serializable(spec) or old["source_sha256"] != hashes:
            raise ValueError("Changed protocol or source: use a fresh output archive")
    else:
        for name in SOURCES:
            dest = output/"source_snapshots"/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
        write_json(path, dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol=spec,
            source_sha256=hashes, command=[sys.executable, *sys.argv],
            versions={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "jax", "jaxlib", "nashopt")}))
    return spec


def make_model(weights, recovery=False, *, effort_weight=.1):
    from game_theory_mpc.blocked_game_mpc import CooperativeWeights
    from game_theory_mpc.cooperative_equilibrium_mpc import CooperativeEquilibriumMPC
    return CooperativeEquilibriumMPC(3, BLOCKS, method="vgne", velocity_weight=weights[1],
        cooperative_weights=CooperativeWeights(weights[0], weights[2], effort_weight), terminal_scale=1., input_bound=2.,
        speed_bound=2.2 if recovery else 2., jerk_bound=20. if recovery else 6.,
        altitude_bounds=(.3, 2.) if recovery else (.35, 1.95), max_iterations=200)


def metrics(times, states, controls, targets, expected_duration, termination, accepted, applied):
    """Metrics have fixed physical units and coefficients, independent of weights."""
    times, states, controls, targets = map(np.asarray, (times, states, controls, targets))
    errors2 = np.sum((states[:, :, :3]-targets[None])**2, axis=2)
    speeds = np.linalg.norm(states[:, :, 3:], axis=2)
    arrived = np.all((errors2 <= .25**2)&(speeds <= .3), axis=1)
    arrival = None
    for i in np.flatnonzero(arrived):
        end = np.searchsorted(times, times[i]+.5-1e-9)
        if end < len(times) and np.all(arrived[i:end+1]):
            arrival = float(times[i])
            break
    squared_sum = errors2.sum(axis=1)
    ise = float(np.sum(.5*(squared_sum[:-1]+squared_sum[1:])*np.diff(times)))
    effort = float(np.sum(np.sum(controls**2, axis=(1, 2))*np.diff(times))) if len(controls) else 0.
    min_d = float(min(np.linalg.norm(states[:, i, :3]-states[:, j, :3], axis=1).min()
        for i in range(3) for j in range(i+1, 3)))
    completed = bool(termination == "update_limit" and abs(times[-1]-expected_duration) < 1e-7)
    success = bool(completed and arrival is not None)
    score = ise/(expected_duration*max(float(errors2[0].sum()), 1e-12))+.05*effort/(expected_duration*3*3*4)
    return dict(completed=completed, success=success, termination=termination, duration_s=float(times[-1]),
        joint_arrival_s=arrival, integrated_position_error_m2s=ise, input_effort_m2_s3=effort,
        minimum_sampled_distance_m=min_d, final_distances_m=np.sqrt(errors2[-1]), final_speeds_mps=speeds[-1],
        score=score if success else None, accepted=int(accepted), applied=int(applied))


def write_attempt(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf8") as stream:
        json.dump(serializable(record), stream, allow_nan=False)
    temporary.replace(path)


def read_attempt(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def rollout(directory, case, weights, updates, *, effort_weight=.1):
    if (directory/"summary.json").exists():
        return read(directory/"summary.json")
    # Incomplete episodes restart deterministically, preserving their old attempts.
    if directory.exists():
        old = directory.with_name(directory.name+"_interrupted_"+datetime.now(timezone.utc).strftime("%H%M%S%f"))
        directory.rename(old)
    directory.mkdir(parents=True)
    write_json(directory/"config.json", dict(case=case, weights=weights, updates=updates, effort_weight=effort_weight))
    model, recovery = make_model(weights, effort_weight=effort_weight), None
    states, targets = np.asarray(case["states"], dtype=float), np.asarray(case["targets"], dtype=float)
    history, controls_history, times = [states.copy()], [], [0.]
    termination, accepted, applied = "update_limit", 0, 0
    for update in range(updates):
        previous = model._previous_controls.copy()
        controls, prediction, info = model.solve(states, targets)
        used_recovery = False
        if not info["feasible"]:
            normal = info
            if recovery is None:
                recovery = make_model(weights, recovery=True, effort_weight=effort_weight)
            recovery._warm, recovery._solved = np.asarray(info["blocked_controls"]).copy(), True
            recovery.commit_controls(previous)
            controls, prediction, info = recovery.solve(states, targets)
            info["tightened_attempt"] = normal
            model._warm = recovery._warm.copy()
            used_recovery = True
        extent = first_interval_extrema(states, controls[:, 0])
        maximum_speed = max(np.linalg.norm(states[:, 3:], axis=1).max(),
            np.linalg.norm(prediction[:, 0, 3:], axis=1).max())
        admissible = bool(info["feasible"] and extent["first_interval_minimum_distance_m"] >= .5-1e-8
            and extent["first_interval_minimum_altitude_m"] >= .3-1e-8
            and extent["first_interval_maximum_altitude_m"] <= 2.+1e-8 and maximum_speed <= 2.5+1e-8)
        write_attempt(directory/"attempts"/f"k{update:03d}.json.gz", dict(update=update,
            states=states, targets=targets, previous_controls=previous, controls=controls,
            prediction=prediction, recovery=used_recovery, applied=admissible, first_interval=extent, **info))
        if not admissible:
            termination = "no_feasible_control"
            break
        next_states = states.copy()
        next_states[:, :3] += .1*states[:, 3:]+.005*controls[:, 0]
        next_states[:, 3:] += .1*controls[:, 0]
        np.testing.assert_allclose(next_states, prediction[:, 0], atol=1e-11)
        states = next_states
        model.commit_controls(controls[:, 0])
        applied += 1
        accepted += info["validation"]["accepted"]
        history.append(states.copy())
        controls_history.append(controls[:, 0].copy())
        times.append(applied*.1)
    np.savez_compressed(directory/"trajectory.npz", times=times, states=history,
        controls=np.asarray(controls_history).reshape(-1, 3, 3), targets=targets)
    result = metrics(times, history, np.asarray(controls_history).reshape(-1, 3, 3), targets,
        updates*.1, termination, accepted, applied)
    write_json(directory/"summary.json", result)
    print(f"{directory.name}: success={result['success']} accepted={accepted}/{applied}", flush=True)
    return result


def train(output, spec):
    frozen = output/"frozen_weights.json"
    if frozen.exists():
        return read(frozen)
    cache = {}
    def evaluate(weights):
        k = key(weights)
        if k not in cache:
            episodes = [rollout(output/"design"/k/c["id"], c, weights, spec["updates"]) for c in spec["cases"]["train"]]
            eligible = all(x["success"] for x in episodes)
            cache[k] = dict(weights=weights, eligible=eligible,
                score=float(np.mean([x["score"] for x in episodes])) if eligible else None, episodes=episodes)
            write_json(output/"design"/k/"evaluation.json", cache[k])
        return cache[k]
    current = evaluate(BASELINE)
    if not current["eligible"]:
        raise RuntimeError("Baseline does not finish the design tasks; retain this archive and inspect before tuning")
    trace, tracking = [key(BASELINE)], None
    for axis, candidates in enumerate(AXES):
        options = [current]
        for value in candidates:
            weights = list(current["weights"])
            weights[axis] = value
            options.append(evaluate(tuple(weights)))
        # Stable sort retains the incumbent on ties.
        current = min(options, key=lambda x: float("inf") if not x["eligible"] else x["score"])
        trace.append(key(current["weights"]))
        if axis == 0:
            tracking = current
    result = dict(baseline=BASELINE, tracking_only=tracking["weights"], full=current["weights"],
        design_scores={k: dict(weights=v["weights"], score=v["score"], eligible=v["eligible"]) for k, v in cache.items()},
        coordinate_trace=trace, frozen_before_test=True,
        test_case_sha256=hashlib.sha256(json.dumps(spec["cases"]["test"], sort_keys=True).encode()).hexdigest())
    write_json(frozen, result)
    return result


def evaluate(output, spec):
    frozen = read(output/"frozen_weights.json")
    for method in ("baseline", "full"):
        for case in spec["cases"]["test"]:
            rollout(output/"test"/method/case["id"], case, frozen[method], spec["updates"])


def physics(output, spec, *, profiles=None):
    from experiments.experiment_blocked_terminal_mpc import run_case
    if profiles is None:
        frozen = read(output/"frozen_weights.json")
        profiles = {method: dict(weights=frozen[method], effort_weight=.1)
                    for method in ("baseline", "full")}
    for case_id in spec["cases"]["physics_ids"]:
        case = next(c for c in spec["cases"]["test"] if c["id"] == case_id)
        override = dict(states=np.asarray(case["states"], dtype=float), targets=np.asarray(case["targets"], dtype=float),
            obstacles=np.empty((0, 3)), pursuers=0, short_horizon=5)
        for method, profile in profiles.items():
            weights = profile["weights"]
            directory = output/"physics"/method/case_id
            if (directory/"summary.json").exists():
                continue
            if directory.exists():
                directory.rename(directory.with_name(case_id+"_interrupted_"+datetime.now(timezone.utc).strftime("%H%M%S%f")))
            directory.mkdir(parents=True)
            args = SimpleNamespace(cooperative_layout="all-moving", blocks=BLOCKS, goal_weight=2., evasion_weight=50.,
                physical_speed_bound=2.5, equilibrium_method="vgne", velocity_weight=weights[1],
                position_weight=weights[0], repulsion_weight=weights[2], effort_weight=profile["effort_weight"],
                terminal_scale=1., input_bound=2.,
                max_iterations=200, response_sweeps=5, speed_bound=2., jerk_bound=6., updates=spec["updates"],
                pid_mode="limited", reserve_retry=True, emergency_jerk_bound=20.)
            run_case("cooperative", "terminal", args, directory, case_override=override)
            trajectory = np.load(directory/"cooperative_terminal"/"trajectory.npz")
            old = read(directory/"cooperative_terminal"/"summary.json")
            paths = sorted((directory/"cooperative_terminal"/"attempts").glob("k*.json"))
            attempts = [read(p) for p in paths]
            result = metrics(trajectory["times"], trajectory["states"], trajectory["accelerations"][1:],
                override["targets"], spec["updates"]*.1, old["termination"],
                sum(a["applied"] and a["validation"]["accepted"] for a in attempts), sum(a["applied"] for a in attempts))
            write_json(directory/"summary.json", result)


def report(output, spec):
    frozen = read(output/"frozen_weights.json")
    sections = {}
    lines = ["# Closed-loop weight transfer", "", "Weights: position / velocity / repulsion; input weight 0.1.", ""]
    for phase, count in (("test", 12), ("physics", 2)):
        lines += [f"## {phase}", "", "| Method | Weights | Success | Arrival median [s] | ISE median [m²s] | Effort median [m²/s³] | Min. distance [m] | Accepted/applied |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
        sections[phase] = {}
        for method in ("baseline", "full"):
            ids = [c["id"] for c in spec["cases"]["test"]] if phase == "test" else spec["cases"]["physics_ids"]
            episodes = [read(output/phase/method/i/"summary.json") for i in ids if (output/phase/method/i/"summary.json").exists()]
            success = [x for x in episodes if x["success"]]
            median = lambda field: float(np.median([x[field] for x in success])) if success else None
            row = dict(recorded=len(episodes), expected=count, success=len(success),
                arrival_median_s=median("joint_arrival_s"), ise_median_m2s=median("integrated_position_error_m2s"),
                effort_median=median("input_effort_m2_s3"), minimum_distance_m=min((x["minimum_sampled_distance_m"] for x in episodes), default=None),
                accepted=sum(x["accepted"] for x in episodes), applied=sum(x["applied"] for x in episodes))
            sections[phase][method] = row
            fmt = lambda x: "—" if x is None else f"{x:.4f}"
            lines.append(f"| {method} | {' / '.join(f'{v:g}' for v in frozen[method])} | {len(success)}/{len(episodes)} | "
                f"{fmt(row['arrival_median_s'])} | {fmt(row['ise_median_m2s'])} | {fmt(row['effort_median'])} | "
                f"{fmt(row['minimum_distance_m'])} | {row['accepted']}/{row['applied']} |")
        lines += ["", "ISE/effort/arrival medians condition on successful episodes; all failures remain in the denominator and raw archive.", ""]
    write_json(output/"results.json", sections)
    (output/"report.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "train", "test", "physics", "report", "all"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()
    from tools.reproduce import writable_output
    writable_output(args.output)
    if args.updates < 1:
        parser.error("updates must be positive")
    with archive_lock(args.output):
        spec = initialize(args.output, args.updates)
        if args.phase == "plan":
            print(json.dumps(serializable(spec), indent=2))
        if args.phase in ("train", "all"):
            train(args.output, spec)
        if args.phase in ("test", "all"):
            evaluate(args.output, spec)
        if args.phase in ("physics", "all"):
            physics(args.output, spec)
        if args.phase in ("report", "all"):
            report(args.output, spec)


if __name__ == "__main__":
    main()
