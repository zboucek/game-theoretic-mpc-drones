#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Check cooperative flight candidates for article VI-A.

Independent JAX derivatives verify NumPy solver outputs, multiplier topology,
active-set residuals and unilateral witnesses, including candidate repairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import nnls
from experiments.audit_blocked_terminal_mpc import close, read, sources
from game_theory_mpc.blocked_game_mpc import BlockedGameMPC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Leave archived validation files unchanged")
    args = parser.parse_args()
    from tools.reproduce import writable_output
    if args.source is not None and not getattr(args, "check_only", False): writable_output(args.source)
    cases = []
    for method in ("gne", "vgne"):
        root = args.source/method
        sources(root)
        assert read(root/"summary.json")["complete"]
        directory = root/"cooperative_terminal"
        config = read(directory/"config.json")
        assert config["num_pursuers"] == 0 and config["hard_pair_separation"]
        models, updates = {}, []
        for path in sorted((directory/"attempts").glob("k*.json")):
            record = read(path)
            key = (tuple(record["constraint_altitude_bounds"]), record["constraint_speed_bound"], record["constraint_jerk_bound"])
            if key not in models:
                models[key] = BlockedGameMPC(config["num_agents"], config["block_lengths"],
                    velocity_weight=config["velocity_weight"], terminal_scale=config["terminal_scale"],
                    input_bound=config["input_bound"], altitude_bounds=key[0], speed_bound=key[1], jerk_bound=key[2])
            model = models[key]
            states, targets = np.asarray(record["states"]), np.asarray(record["targets"])
            controls = np.asarray(record["blocked_controls"]).ravel()
            model.commit_controls(record["previous_controls"])
            parameters = model.parameters(states, targets)
            values = model.evaluate_numpy(controls, states, targets)
            close(values["prediction"], record["prediction"])
            close(values["costs"], record["costs"])
            gradient = np.asarray(model.potential_gradient(controls, parameters))
            jacobian = np.vstack([np.asarray(model.constraint_jacobian(controls, parameters)),
                                  -np.eye(model.nvar), np.eye(model.nvar)])
            g = np.r_[values["g"], -model.input_bound-controls, controls-model.input_bound]
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
            if method == "vgne":
                close(multipliers[:, :model.ng], np.broadcast_to(multipliers[0, :model.ng], (model.num_agents, model.ng)))
            # Unequal individual multipliers alone do not disprove vGNE:
            # also test whether a common nonnegative multiplier vector fits.
            active = g >= -1e-6
            common = np.zeros(len(g))
            if active.any():
                common[active], _ = nnls(jacobian[active].T, -gradient, maxiter=10000)
            common_fit = float(np.linalg.norm(gradient+jacobian.T@common))
            checked_deviations = 0
            for dev in record["deviations"]:
                player = dev["player"]
                changed = controls.reshape(model.num_agents, model.blocking.blocks, 3).copy()
                changed[player] = dev["own_blocked_controls"]
                independent = model.evaluate_numpy(changed.ravel(), states, targets)
                improvement = float(values["costs"][player]-independent["costs"][player])
                feasible = bool(independent["g"].max() <= 1e-7 and np.abs(changed).max() <= model.input_bound+1e-8)
                close(improvement, dev["improvement"])
                assert feasible == dev["feasible"]
                assert (feasible and improvement > dev["threshold"]) == dev["refutes_local_candidate"]
                checked_deviations += 1
            checked_repairs = 0
            for rejected in record.get("repair_history", []):
                rejected_controls = np.asarray(rejected["blocked_controls"])
                baseline = model.evaluate_numpy(rejected_controls.ravel(), states, targets)
                close(baseline["costs"], rejected["costs"])
                witness = rejected["witness"]
                changed = rejected_controls.copy()
                changed[witness["player"]] = witness["own_blocked_controls"]
                improved = model.evaluate_numpy(changed.ravel(), states, targets)
                improvement = float(baseline["costs"][witness["player"]]-improved["costs"][witness["player"]])
                close(improvement, witness["improvement"])
                assert improved["g"].max() <= 1e-7 and np.abs(changed).max() <= model.input_bound+1e-8
                assert improvement > witness["threshold"]
                # In this cooperative exact-potential game the unilateral
                # improvement must also reduce the joint potential equally.
                close(baseline["potential"]-improved["potential"], improvement)
                checked_repairs += 1
            independently_feasible = bool(np.isfinite(controls).all() and np.isfinite(g).all() and g.max() <= 1e-7)
            assert independently_feasible == record["feasible"]
            accepted = bool(independently_feasible and norm <= 1e-5 and multipliers.min() >= -1e-7
                            and not any(dev["refutes_local_candidate"] for dev in record["deviations"]))
            assert accepted == record["validation"]["accepted"]
            updates.append(dict(update=record["update"], applied=record["applied"], accepted=accepted,
                kkt_norm=norm, common_multiplier_stationarity_fit=common_fit,
                collision_participant_multiplier_difference=record["validation"]["collision_participant_multiplier_difference"],
                checked_deviations=checked_deviations, checked_repairs=checked_repairs))
        case = dict(method=method, passed=True, updates=updates,
            applied=sum(u["applied"] for u in updates), accepted_applied=sum(u["applied"] and u["accepted"] for u in updates),
            maximum_recorded_kkt_norm=max(u["kkt_norm"] for u in updates),
            checked_repairs=sum(u["checked_repairs"] for u in updates),
            maximum_common_multiplier_fit=max(u["common_multiplier_stationarity_fit"] for u in updates))
        cases.append(case)
        if not args.check_only:
            (root/"equilibrium_validation.json").write_text(json.dumps(case, indent=2)+"\n")
            snapshot = root/"audit_sources"/Path(__file__).name
            snapshot.parent.mkdir(exist_ok=True)
            snapshot.write_bytes(Path(__file__).read_bytes())
    result = dict(passed=True, cases=cases,
        scope="Independent JAX derivatives against NumPy primal/active-set solver, recorded multiplier topology and unilateral witnesses. No global Nash certificate.")
    if not args.check_only:
        (args.source/"equilibrium_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=True, cases=[{k:v for k,v in case.items() if k != 'updates'} for case in cases])))


if __name__ == "__main__":
    main()
