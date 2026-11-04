#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Independent candidate checks for the weight study (article IV-B).

Audit rebuilds each weight/recovery model and checks costs, constraints,
multipliers and unilateral responses. tools/results.py also checks trajectories."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.cooperative_weight_study import read, read_attempt, metrics, BLOCKS
from experiments.equilibrium_selection.archive import digest, write_json
from experiments.equilibrium_mpc_execution import first_interval_extrema
from game_theory_mpc.blocked_game_mpc import BlockedGameMPC, CooperativeWeights


def close(a, b):
    np.testing.assert_allclose(a, b, atol=2e-7, rtol=1e-7)


def check_default_compatibility():
    from experiments.experiment_blocked_terminal_mpc import scenario
    case = scenario('cooperative', 'all-moving')
    a = BlockedGameMPC(3, BLOCKS, velocity_weight=1., terminal_scale=1.)
    b = BlockedGameMPC(3, BLOCKS, velocity_weight=1., terminal_scale=1., cooperative_weights=CooperativeWeights())
    u = np.zeros(a.nvar)
    x, y = (m.evaluate_numpy(u, case['states'], case['targets']) for m in (a, b))
    close(x['costs'], y['costs'])
    close(x['g'], y['g'])
    return dict(checked=True)


class Audit:
    def __init__(self):
        self.models = {}
        self.records, self.deviations, self.repairs = 0, 0, 0

    def tightened(self, record, weights, effort_weight=.1):
        normal = dict(record["tightened_attempt"], states=record["states"], targets=record["targets"],
            previous_controls=record["previous_controls"])
        # A retry archives the normal solver's complete info, without duplicate
        # state/prediction arrays. Independently reconstruct those arrays.
        model = BlockedGameMPC(3, BLOCKS, velocity_weight=weights[1], terminal_scale=1.,
            cooperative_weights=CooperativeWeights(weights[0], weights[2], effort_weight), input_bound=2., speed_bound=2., jerk_bound=6.)
        values = model.evaluate_numpy(np.asarray(normal["blocked_controls"]).ravel(), normal["states"], normal["targets"])
        normal.update(controls=values["full_controls"], prediction=values["prediction"])
        return self.candidate(normal, weights, effort_weight=effort_weight)

    def candidate(self, record, weights, recovery=False, effort_weight=.1):
        identity = (*weights, recovery, effort_weight)
        if identity not in self.models:
            self.models[identity] = BlockedGameMPC(3, BLOCKS, velocity_weight=weights[1],
                cooperative_weights=CooperativeWeights(weights[0], weights[2], effort_weight), terminal_scale=1.,
                input_bound=2., speed_bound=2.2 if recovery else 2., jerk_bound=20. if recovery else 6.,
                altitude_bounds=(.3, 2.) if recovery else (.35, 1.95))
        model = self.models[identity]
        model.commit_controls(record["previous_controls"])
        states, targets = np.asarray(record["states"]), np.asarray(record["targets"])
        controls = np.asarray(record["blocked_controls"]).ravel()
        values = model.evaluate_numpy(controls, states, targets)
        close(values["prediction"], record["prediction"])
        close(values["costs"], record["costs"])
        close(values["full_controls"], record["controls"])
        for name in values["terms"]:
            close(values["terms"][name], record["cost_terms"][name])
        parameters = model.parameters(states, targets)
        gradient = np.asarray(model.potential_gradient(controls, parameters))
        jac = np.vstack([np.asarray(model.constraint_jacobian(controls, parameters)), -np.eye(model.nvar), np.eye(model.nvar)])
        g = np.r_[values["g"], -model.input_bound-controls, controls-model.input_bound]
        lam = np.asarray(record["multipliers"])
        assert lam.shape == (3, len(g))
        close(lam, np.broadcast_to(lam[0], lam.shape))
        stat, fb = [], []
        for i in range(3):
            own = slice(i*model.width, (i+1)*model.width)
            stat.extend(gradient[own]+jac[:, own].T@lam[i])
            relevant = np.max(np.abs(jac[:, own]), axis=1) > 1e-12
            fb.extend(np.hypot(lam[i, relevant], g[relevant])-lam[i, relevant]+g[relevant])
        norm = float(np.linalg.norm(np.r_[stat, fb]))
        close(norm, record["validation"]["kkt_norm"])
        feasible = bool(np.isfinite(controls).all() and np.isfinite(g).all() and g.max() <= 1e-7)
        assert feasible == record["feasible"]
        for dev in record["deviations"]:
            changed = controls.reshape(3, len(BLOCKS), 3).copy()
            changed[dev["player"]] = dev["own_blocked_controls"]
            other = model.evaluate_numpy(changed.ravel(), states, targets)
            improvement = values["costs"][dev["player"]]-other["costs"][dev["player"]]
            close(improvement, dev["improvement"])
            valid = bool(other["g"].max() <= 1e-7 and np.abs(changed).max() <= model.input_bound+1e-8)
            assert valid == dev["feasible"]
            assert bool(valid and improvement > dev["threshold"]) == dev["refutes_local_candidate"]
            self.deviations += 1
        for rejected in record.get("repair_history", []):
            original = np.asarray(rejected["blocked_controls"])
            before = model.evaluate_numpy(original.ravel(), states, targets)
            close(before["costs"], rejected["costs"])
            witness = rejected["witness"]
            changed = original.copy()
            changed[witness["player"]] = witness["own_blocked_controls"]
            after = model.evaluate_numpy(changed.ravel(), states, targets)
            improvement = before["costs"][witness["player"]]-after["costs"][witness["player"]]
            close(improvement, witness["improvement"])
            close(before["potential"]-after["potential"], improvement)
            assert after["g"].max() <= 1e-7 and np.abs(changed).max() <= model.input_bound+1e-8
            assert improvement > witness["threshold"]
            self.repairs += 1
        accepted = bool(feasible and norm <= 1e-5 and lam.min() >= -1e-7
            and not any(d["refutes_local_candidate"] for d in record["deviations"]))
        assert accepted == record["validation"]["accepted"]
        self.records += 1
        return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    from tools.reproduce import writable_output
    if args.source is not None and not getattr(args, "check_only", False): writable_output(args.source)
    root = args.source
    manifest, frozen = read(root/"manifest.json"), read(root/"frozen_weights.json")
    spec = manifest["protocol"]
    for name, sha in manifest["source_sha256"].items():
        assert digest(root/"source_snapshots"/name) == sha
        assert digest(ROOT/name) == sha, f"Audit requires the archived producer/model sources: {name}"
    audit, checked = Audit(), []
    directories = sorted((root/"design").glob("*/design_*/config.json"))
    directories += sorted((root/"test").glob("*/*/config.json"))
    expected = len(frozen["design_scores"])*len(spec["cases"]["train"])+36
    assert len(directories) == expected
    for config_path in directories:
        directory, config = config_path.parent, read(config_path)
        weights = config["weights"]
        attempts = [read_attempt(p) for p in sorted((directory/"attempts").glob("k*.json.gz"))]
        applied, accepted, min_continuous = 0, 0, float("inf")
        traj = np.load(directory/"trajectory.npz")
        close(traj["states"][0], config["case"]["states"])
        for i, r in enumerate(attempts):
            assert r["update"] == i
            close(r["states"], traj["states"][applied])
            close(r["previous_controls"], np.zeros((3, 3)) if i == 0 else traj["controls"][applied-1])
            if "tightened_attempt" in r:
                audit.tightened(r, weights)
            ok = audit.candidate(r, weights, r["recovery"])
            extent = first_interval_extrema(r["states"], np.asarray(r["controls"])[:, 0])
            for name in extent:
                close(extent[name], r["first_interval"][name])
            if r["applied"]:
                assert r["feasible"] and extent["first_interval_minimum_distance_m"] >= .5-1e-8
                min_continuous = min(min_continuous, extent["first_interval_minimum_distance_m"])
                close(r["prediction"][0][0], traj["states"][applied+1, 0])
                close(np.asarray(r["controls"])[:, 0], traj["controls"][applied])
                applied += 1
                accepted += ok
        assert len(traj["times"]) == applied+1
        summary = read(directory/"summary.json")
        recalculated = metrics(traj["times"], traj["states"], traj["controls"], traj["targets"],
            spec["updates"]*.1, summary["termination"], accepted, applied)
        for name in recalculated:
            if recalculated[name] is None or isinstance(recalculated[name], (str, bool)):
                assert summary[name] == recalculated[name]
            else:
                close(summary[name], recalculated[name])
        # Independently check exact double-integrator transitions for all players.
        close(traj["states"][1:, :, :3], traj["states"][:-1, :, :3]+.1*traj["states"][:-1, :, 3:]+.005*traj["controls"])
        close(traj["states"][1:, :, 3:], traj["states"][:-1, :, 3:]+.1*traj["controls"])
        checked.append(dict(path=str(directory.relative_to(root)), applied=applied, accepted=accepted,
            minimum_continuous_distance_m=min_continuous))
        print(f"checked {directory.relative_to(root)}", flush=True)
    for method in ("baseline", "full"):
        for case_id in spec["cases"]["physics_ids"]:
            directory = root/"physics"/method/case_id
            summary = read(directory/"summary.json")
            inner = directory/"cooperative_terminal"
            attempts = [read(p) for p in sorted((inner/"attempts").glob("k*.json"))]
            accepted, applied = 0, 0
            traj = np.load(inner/"trajectory.npz")
            for i, r in enumerate(attempts):
                close(r["states"], traj["states"][i*12])
                if "tightened_attempt" in r:
                    audit.tightened(r, frozen[method])
                recovery = r["constraint_jerk_bound"] == 20.
                ok = audit.candidate(r, frozen[method], recovery)
                if r["applied"]:
                    accepted += ok
                    applied += 1
            assert summary["accepted"] == accepted and summary["applied"] == applied
            assert np.isfinite(traj["states"]).all()
            assert np.abs(traj["accelerations"]).max() <= 2.+1e-7
            if summary["completed"]:
                assert traj["states"][:, :, 2].min() >= .3
                assert traj["states"][:, :, 2].max() <= 2.
                assert np.linalg.norm(traj["states"][:, :, 3:], axis=2).max() <= 2.5
                assert summary["minimum_sampled_distance_m"] >= .5
            recomputed = metrics(traj["times"], traj["states"], traj["accelerations"][1:], traj["targets"],
                spec["updates"]*.1, summary["termination"], accepted, applied)
            for name in recomputed:
                if recomputed[name] is None or isinstance(recomputed[name], (str, bool)):
                    assert summary[name] == recomputed[name]
                else:
                    close(summary[name], recomputed[name])
            checked.append(dict(path=str(directory.relative_to(root)), applied=summary["applied"], accepted=summary["accepted"]))
    write_json(root/"audit.json", dict(passed=True, episodes=len(checked), candidate_records=audit.records,
        checked_deviations=audit.deviations, checked_repairs=audit.repairs, cases=checked,
        audit_source_sha256=digest(Path(__file__))))
    write_json(root/"default_compatibility.json", check_default_compatibility())
    snapshot = root/"audit_sources"/Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in ("artifacts.json", ".study.lock"))
    write_json(root/"artifacts.json", {str(p.relative_to(root)): digest(p) for p in files})
    print(f"PASS: {len(checked)} episodes, {audit.records} candidates", flush=True)


if __name__ == "__main__":
    main()
