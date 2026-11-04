#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Matched controller timing study for article IV-C.

design defines agent/horizon sweeps and full/blocked pairs. Each serial worker
restores identical inputs, discards one solve and records repeated calls.
audit_current_scalability checks records after measurement."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.equilibrium_selection.archive import write_json
from game_theory_mpc.blocked_game_mpc import AdversarialWeights, BlockedGameMPC
from game_theory_mpc.cooperative_equilibrium_mpc import CooperativeEquilibriumMPC

SOURCES = ("experiments/benchmark_current_scalability.py", "experiments/equilibrium_selection/archive.py",
           "src/game_theory_mpc/blocked_game_mpc.py", "src/game_theory_mpc/cooperative_equilibrium_mpc.py")
AGENTS = (2, 3, 4, 5, 6, 8, 10)
HORIZONS = (10, 15, 20, 30, 40)
METHODS = ("gne", "vgne", "adversarial_br")


def read(path):
    return json.loads(path.read_text())


def block_lengths(horizon, parameterization):
    if parameterization == "full":
        return (1,)*horizon
    if horizon < 10:
        raise ValueError("The six-block schedule needs H >= 10")
    remaining = horizon-8
    return (1, 1, 2, 4, remaining//2, remaining-remaining//2)


def scenario(method, n):
    if method != "adversarial_br":
        x, targets = [], []
        for i in range(n):
            cell, role = divmod(i, 3)
            y = 6.*cell
            x.append(([-.3, y, 1., .2, 0., 0.], [.3, y, 1., -.1, 0., 0.], [0., y+2, 1., 0., 0., 0.])[role])
            targets.append(([10., y, 1.], [-6., y, 1.], [0., y-2, 1.])[role])
        return np.asarray(x), np.asarray(targets), 0, np.empty((0, 3))
    pursuers = max(1, min(n-1, round(.6*n)))
    evaders = n-pursuers
    x = []
    for i in range(pursuers):
        cell, role = divmod(i, 3)
        value = ([2., 2., 1., 0., 0., 0.], [2., -2., 1., 0., 0., 0.], [-2., 2., 1., 0., 0., 0.])[role].copy()
        value[1] += 6.*cell
        x.append(value)
    targets = []
    for i in range(evaders):
        cell, role = divmod(i, 2)
        x.append([-1. if role == 0 else 1., 6.*cell, 1., 0., 0., 0.])
        targets.append([3. if role == 0 else -3., 6.*cell+(1. if role == 0 else -1.), 1.])
    obstacles = np.asarray([[0., 1., 1.], [0., -1., 1.], [1.5, .5, 1.], [-1.5, -.5, 1.]])
    return np.asarray(x), np.asarray(targets), pursuers, obstacles


def initial_controls(method, n, lengths):
    seed = np.zeros((n, len(lengths), 3))
    active = np.cumsum((0, *lengths[:-1])) < 8
    if method != "adversarial_br":
        for i in range(n):
            role = i % 3
            if role < 2:
                seed[i, active, 0] = -.6 if role == 0 else .6
                seed[i, active, 1] = .3 if role == 0 else -.3
    return seed


def design(suite):
    cases = []
    for method in METHODS:
        base = 5 if method == "adversarial_br" else 3
        sizes = {(n, 20) for n in AGENTS} | {(base, h) for h in HORIZONS}
        for n, h in sorted(sizes):
            cases.append(dict(method=method, agents=n, horizon=h, parameterization="blocked",
                              sweeps=[key for key, ok in (("agents", h == 20), ("horizon", n == base)) if ok]))
        for h in (10, 20, 30):
            cases.append(dict(method=method, agents=base, horizon=h, parameterization="full", sweeps=["ablation"]))
    for row in cases:
        row["id"] = f"{row['method']}_n{row['agents']}_h{row['horizon']}_{row['parameterization']}"
    if suite == "pilot":
        wanted = {f"{m}_n{5 if m == 'adversarial_br' else 3}_h10_{p}" for m in METHODS for p in ("full", "blocked")}
        cases = [r for r in cases if r["id"] in wanted]
    return cases


def verify_sources(manifest):
    for name, expected in manifest["source_sha256"].items():
        assert sha256((ROOT/name).read_bytes()).hexdigest() == expected, f"Changed source: {name}"


def worker(output, case_id):
    manifest = read(output/"manifest.json")
    verify_sources(manifest)
    case, = [r for r in manifest["cases"] if r["id"] == case_id]
    method, n, h = case["method"], case["agents"], case["horizon"]
    lengths = block_lengths(h, case["parameterization"])
    states, targets, pursuers, obstacles = scenario(method, n)
    seed = initial_controls(method, n, lengths)
    started = perf_counter()
    options = dict(num_pursuers=pursuers, dt=.1, obstacles=obstacles,
                   velocity_weight=5. if pursuers else 1., terminal_scale=1., input_bound=2.,
                   altitude_bounds=(.35, 1.95), speed_bound=2., jerk_bound=6.,
                   max_iterations=200, response_sweeps=5)
    model = (BlockedGameMPC(n, lengths, adversarial_weights=AdversarialWeights(goal=100., evasion=10.), **options)
             if pursuers else CooperativeEquilibriumMPC(n, lengths, method=method, max_sweeps=30,
                                                       repair_deviations=True, **options))
    construction = perf_counter()-started
    model.commit_controls(np.zeros((n, 3)))
    initial = model.evaluate_numpy(seed.ravel(), states, targets)
    assert initial["g"].max() <= 1e-7 and np.abs(seed).max() <= model.input_bound
    root = output/"cases"/case_id
    root.mkdir(parents=True, exist_ok=True)
    config = dict(scenario="adversarial" if pursuers else "cooperative", variant="terminal",
        num_agents=n, num_pursuers=pursuers, horizon=h, block_lengths=lengths, control_blocks=len(lengths),
        initial_states=states, targets=targets, obstacles=obstacles, dt=model.dt,
        velocity_weight=model.velocity_weight, terminal_scale=model.terminal_scale, terminal_extra_matrix=model.P,
        hard_pair_separation=model.hard_pairs, planned_separation_m=model.separation,
        planned_altitude_bounds=model.altitude_bounds, input_bound=model.input_bound,
        speed_bound=model.speed_bound, jerk_bound=model.jerk_bound,
        adversarial_weights=dict(goal=100., evasion=10., pursuit=10., repulsion=2., obstacle=5., effort=.1),
        method=method, parameterization=case["parameterization"], primal_variables=model.nvar,
        shared_inequalities=model.ng, initial_blocked_controls=seed, previous_controls=model._previous_controls,
        max_iterations=200, max_sweeps=30 if not pursuers else None, response_sweeps=5,
        initialization="Same feasible warm-start controls and zero previous applied input restored before every call")
    write_json(root/"config.json", config)
    preparation = model.prepare(states, targets)
    write_json(root/"preparation.json", dict(construction_seconds=construction, preparation_seconds=preparation))
    completed = sorted((root/"samples").glob("timed_*.json"))
    warmups = len(list((root/"samples").glob("warmup_*.json")))
    flags = [(True, warmups)] + [(False, k) for k in range(len(completed), manifest["repeats"])]
    for discarded, repeat in flags:
        model._warm, model._solved = seed.copy(), True
        model.commit_controls(np.zeros((n, 3)))
        if not pursuers:
            model._repair_depth = 0
        label = f"{'warmup' if discarded else 'timed'}_{repeat:02d}"
        write_json(root/"active_call.json", dict(label=label, started_utc=datetime.now(timezone.utc).isoformat()))
        started = perf_counter()
        controls, prediction, info = model.solve(states, targets)
        elapsed = perf_counter()-started
        np.testing.assert_array_equal(model._previous_controls, np.zeros((n, 3)))
        if not pursuers:
            np.testing.assert_array_equal(info["initial_blocked_controls"], seed)
            assert model._repair_depth == 0
        record = dict(info, states=states, targets=targets, controls=controls, prediction=prediction,
            previous_controls=model._previous_controls, initial_blocked_controls=seed,
            constraint_altitude_bounds=model.altitude_bounds, constraint_speed_bound=model.speed_bound,
            constraint_jerk_bound=model.jerk_bound, warmup=discarded, repeat=repeat, external_wall_seconds=elapsed,
            case_id=case_id, applied=False)
        write_json(root/"samples"/f"{label}.json", record)
        print(f"{case_id} {label}: {elapsed:.4f}s feasible={info['feasible']} equilibrium={info.get('validation', {}).get('accepted')}", flush=True)
    verify_sources(manifest)
    write_json(root/"summary.json", dict(complete=True, timed_samples=manifest["repeats"], case=case))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("pilot", "full"), default="full")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--case-timeout", type=float, default=600.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    from tools.reproduce import writable_output
    writable_output(args.output)
    if args.worker:
        worker(args.output, args.worker)
        return
    if args.repeats < 1 or args.case_timeout <= 0:
        parser.error("Positive repeat count and worker timeout required")
    output = args.output.resolve()
    if args.resume:
        manifest = read(output/"manifest.json")
        assert (manifest["suite"], manifest["repeats"], manifest["case_timeout_seconds"]) == (args.suite, args.repeats, args.case_timeout)
        verify_sources(manifest)
    else:
        output.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for name in SOURCES:
            content = (ROOT/name).read_bytes()
            snapshot = output/"source_snapshots"/name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(content)
            hashes[name] = sha256(content).hexdigest()
        cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        manifest = dict(schema=1, suite=args.suite, created_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=hashes, git_head=None,
            command=[sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
            library_versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "jax", "jaxlib")},
            machine=dict(cpu=cpu, logical_cpus=os.cpu_count(), platform=platform.platform(), blas_threads=1),
            cases=design(args.suite), repeats=args.repeats, case_timeout_seconds=args.case_timeout,
            protocol=dict(timing="Serial complete controller solves, including built-in validation/repair; model construction and JAX preparation excluded. One discarded solve per fresh worker, then repeated identical inputs.",
                initialization="Warm-start execution path. Cooperative: feasible common braking/lateral controls for first 8 nodes, then zero; adversarial: zero controls. Previous applied controls zero. State reset before every call. Full and blocked use identical expanded initial controls.",
                blocking="For H>=10: (1,1,2,4,floor((H-8)/2),ceil((H-8)/2)). Full ablation: H individual controls, identical costs and all-node bounds.",
                scenarios="Cooperative N=3 exactly matches the all-moving flight initial state; replicate three-agent conflict units 6 m apart. Adversarial N=5 matches the selected flight; pursuer/evader role groups repeat 6 m apart, with the same four fixed obstacles.",
                quality="Cooperative acceptance is the existing local feasibility/KKT/deviation test. Adversarial feasibility and sweep change are reported separately; no Nash acceptance certificate.",
                scope="Current GNE/vGNE and five-sweep adversarial controllers. Not the frozen dense-root, static potential study or linear laboratory tracker. Timings are fixed-state workloads, not a real-time flight guarantee."))
        write_json(output/"manifest.json", manifest)
    started = perf_counter()
    results = []
    for case in manifest["cases"]:
        root = output/"cases"/case["id"]
        if (root/"summary.json").exists() and read(root/"summary.json")["complete"]:
            status = "complete"
            seconds = 0.
        else:
            root.mkdir(parents=True, exist_ok=True)
            verify_sources(manifest)
            before = perf_counter()
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--worker", case["id"]]
            with (root/"worker.log").open("a") as log:
                try:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.case_timeout)
                    status = "complete" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    log.write(f"Case worker exceeded {args.case_timeout} s\n")
            seconds = perf_counter()-before
        results.append(dict(**case, status=status, worker_wall_seconds=seconds,
                            timed_samples=len(list((root/"samples").glob("timed_*.json")))))
        write_json(output/"summary.json", dict(complete=False, cases=results, elapsed_seconds=perf_counter()-started))
        print(f"{len(results)}/{len(manifest['cases'])} {case['id']}: {status} ({seconds:.1f}s)", flush=True)
    verify_sources(manifest)
    write_json(output/"summary.json", dict(complete=True, cases=results, elapsed_seconds=perf_counter()-started))


if __name__ == "__main__":
    main()
