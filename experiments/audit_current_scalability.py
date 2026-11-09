#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Check every warmup/timed record and matched pair for article V-B.

Cooperative numerical acceptance and adversarial model feasibility are
separate outcomes. Validation runs after measurement and is not timed."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.audit_blocked_terminal_mpc import close, evaluate, read
from game_theory_mpc.blocked_game_mpc import AdversarialWeights, BlockedGameMPC


def audit_sample(model, config, record):
    states, targets = np.asarray(record["states"]), np.asarray(record["targets"])
    np.testing.assert_array_equal(states, config["initial_states"])
    np.testing.assert_array_equal(targets, config["targets"])
    controls = np.asarray(record["blocked_controls"], dtype=float).ravel()
    model.commit_controls(record["previous_controls"])
    np.testing.assert_array_equal(record["initial_blocked_controls"], config["initial_blocked_controls"])
    np.testing.assert_array_equal(record["previous_controls"], config["previous_controls"])
    assert record["external_wall_seconds"] >= record["compute_time"] > 0
    assert not record["applied"]
    if not np.isfinite(controls).all():
        assert not record["feasible"] and not record.get("validation", {}).get("accepted", False)
        return dict(model_feasible=False, numerical_accepted=False if not model.adversarial else None,
                    nonfinite=True, kkt_norm=None)
    values = model.evaluate_numpy(controls, states, targets)
    g = np.r_[values["g"], -model.input_bound-controls, controls-model.input_bound]
    bound_tolerance = 1e-8 if model.adversarial else 1e-7
    native_feasible = bool(values["g"].max() <= 1e-7 and np.abs(controls).max() <= model.input_bound+bound_tolerance)
    assert native_feasible == record["feasible"]
    # The existing scalar recurrence audit uses a stricter 1e-8 box tolerance.
    # Check that policy separately from the controller's recorded native flag.
    strict_record = dict(record, feasible=bool(values["g"].max() <= 1e-7 and np.abs(controls).max() <= model.input_bound+1e-8))
    _, strict_feasible, worst = evaluate(strict_record, config)
    close(worst, values["g"].max())
    row = dict(model_feasible=native_feasible, strict_box_feasible=strict_feasible,
               maximum_constraint_violation=float(max(0., g.max())), nonfinite=False,
               numerical_accepted=None, kkt_norm=None, deviations=0, repairs=0)
    if model.adversarial:
        row["response_change"] = record["response_change"]
        assert np.isfinite(record["response_change"]) and record["response_change"] >= 0
        return row
    parameters = model.parameters(states, targets)
    gradient = np.asarray(model.potential_gradient(controls, parameters))
    jacobian = np.vstack([np.asarray(model.constraint_jacobian(controls, parameters)),
                          -np.eye(model.nvar), np.eye(model.nvar)])
    multipliers = np.asarray(record["multipliers"])
    assert multipliers.shape == (model.num_agents, len(g))
    stationarity, complementarity = [], []
    for player in range(model.num_agents):
        own = slice(player*model.width, (player+1)*model.width)
        stationarity.extend(gradient[own]+jacobian[:, own].T@multipliers[player])
        relevant = np.abs(jacobian[:, own]).max(axis=1) > 1e-12
        lam, constraints = multipliers[player, relevant], g[relevant]
        complementarity.extend(np.hypot(lam, constraints)-lam+constraints)
    norm = float(np.linalg.norm(np.r_[stationarity, complementarity]))
    close(norm, record["validation"]["kkt_norm"])
    close(row["maximum_constraint_violation"], record["validation"]["maximum_constraint_violation"])
    if config["method"] == "vgne":
        close(multipliers[:, :model.ng], np.broadcast_to(multipliers[0, :model.ng], (model.num_agents, model.ng)))
    refutations = []
    for deviation in record["deviations"]:
        changed = controls.reshape(model.num_agents, model.blocking.blocks, 3).copy()
        changed[deviation["player"]] = deviation["own_blocked_controls"]
        candidate = model.evaluate_numpy(changed.ravel(), states, targets)
        improvement = float(values["costs"][deviation["player"]]-candidate["costs"][deviation["player"]])
        feasible = bool(candidate["g"].max() <= 1e-7 and np.abs(changed).max() <= model.input_bound+1e-8)
        close(improvement, deviation["improvement"])
        assert feasible == deviation["feasible"]
        refuted = bool(feasible and improvement > deviation["threshold"])
        assert refuted == deviation["refutes_local_candidate"]
        refutations.append(refuted)
        row["deviations"] += 1
    for rejected in record.get("repair_history", []):
        before_controls = np.asarray(rejected["blocked_controls"])
        before = model.evaluate_numpy(before_controls.ravel(), states, targets)
        close(before["costs"], rejected["costs"])
        witness = rejected["witness"]
        changed = before_controls.copy()
        changed[witness["player"]] = witness["own_blocked_controls"]
        after = model.evaluate_numpy(changed.ravel(), states, targets)
        improvement = float(before["costs"][witness["player"]]-after["costs"][witness["player"]])
        close(improvement, witness["improvement"])
        assert after["g"].max() <= 1e-7 and np.abs(changed).max() <= model.input_bound+1e-8
        assert improvement > witness["threshold"]
        close(before["potential"]-after["potential"], improvement)
        row["repairs"] += 1
    accepted = bool(native_feasible and norm <= 1e-5 and multipliers.min() >= -1e-7 and not any(refutations))
    assert accepted == record["validation"]["accepted"]
    row.update(numerical_accepted=accepted, kkt_norm=norm)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    from tools.reproduce import writable_output
    if args.source is not None and not getattr(args, "check_only", False): writable_output(args.source)
    manifest, summary = read(args.source/"manifest.json"), read(args.source/"summary.json")
    assert summary["complete"], "Finish serial measurement before the audit"
    cases, paired, raw_hashes, config_hashes = [], {}, {}, {}
    for case in summary["cases"]:
        directory = args.source/"cases"/case["id"]
        samples = sorted((directory/"samples").glob("*.json"))
        if not samples:
            cases.append(dict(case_id=case["id"], worker_status=case["status"], checked_samples=0, samples=[]))
            continue
        config = read(directory/"config.json")
        config_hashes[str((directory/"config.json").relative_to(args.source))] = sha256((directory/"config.json").read_bytes()).hexdigest()
        assert (config["method"], config["num_agents"], config["horizon"], config["parameterization"]) == (case["method"], case["agents"], case["horizon"], case["parameterization"])
        model = BlockedGameMPC(config["num_agents"], config["block_lengths"],
            num_pursuers=config["num_pursuers"], obstacles=config["obstacles"], dt=config["dt"],
            velocity_weight=config["velocity_weight"], terminal_scale=config["terminal_scale"],
            input_bound=config["input_bound"], altitude_bounds=tuple(config["planned_altitude_bounds"]),
            separation=config["planned_separation_m"], speed_bound=config["speed_bound"], jerk_bound=config["jerk_bound"],
            adversarial_weights=AdversarialWeights(**config["adversarial_weights"]))
        assert model.nvar == config["primal_variables"] and model.ng == config["shared_inequalities"]
        key = (config["method"], config["num_agents"], config["horizon"])
        paired.setdefault(key, {})[config["parameterization"]] = config
        checked = []
        for path in samples:
            record = read(path)
            row = audit_sample(model, config, record)
            checked.append(dict(file=path.name, warmup=record["warmup"], repeat=record["repeat"], **row))
            raw_hashes[str(path.relative_to(args.source))] = sha256(path.read_bytes()).hexdigest()
        cases.append(dict(case_id=case["id"], worker_status=case["status"], checked_samples=len(checked), samples=checked))
        print(f"Audited {case['id']}: {len(checked)} samples", flush=True)
    pairs = 0
    for pair in paired.values():
        if set(pair) != {"full", "blocked"}:
            continue
        full, blocked = pair["full"], pair["blocked"]
        a = np.repeat(full["initial_blocked_controls"], full["block_lengths"], axis=1)
        b = np.repeat(blocked["initial_blocked_controls"], blocked["block_lengths"], axis=1)
        np.testing.assert_array_equal(a, b)
        for field in ("initial_states", "targets", "obstacles", "previous_controls", "terminal_extra_matrix",
                      "velocity_weight", "terminal_scale", "input_bound", "speed_bound", "jerk_bound", "planned_altitude_bounds"):
            np.testing.assert_array_equal(full[field], blocked[field])
        pairs += 1
    sources = ("experiments/audit_current_scalability.py", "experiments/audit_blocked_terminal_mpc.py")
    report = dict(passed=True, checked_samples=sum(c["checked_samples"] for c in cases),
        matched_full_blocked_pairs=pairs, cases=cases, raw_sha256=raw_hashes, config_sha256=config_hashes,
        audit_source_sha256={name: sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        scope="Explicit scalar recurrence/cost/constraint audit, independent JAX KKT/multiplier/deviation/repair checks for cooperative modes, native feasibility for adversarial controls. No global Nash certificate.")
    if not args.check_only:
        write = args.source/"validation.json"
        write.write_text(json.dumps(report, indent=2)+"\n")
        for name in sources:
            target = args.source/"audit_sources"/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT/name).read_bytes())
    print(json.dumps({k:report[k] for k in ("passed", "checked_samples", "matched_full_blocked_pairs")}))


if __name__ == "__main__":
    main()
