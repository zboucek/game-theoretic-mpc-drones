#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Independent dynamics, scalar-cost and flight-telemetry checks (V-A/B).

evaluate reconstructs predictions explicitly; interval_extrema checks between
nodes, and pid_telemetry checks recorded force/reference consistency."""
import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are


def read(path):
    return json.loads(path.read_text())


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-10)


def sources(root):
    # Reference integrity is checked by the public entry point before analysis.
    # New experiment outputs are checked numerically below.
    return None


def evaluate(record, config):
    """Independent explicit time stepping and scalar stage-cost formulas."""
    lengths = config["block_lengths"]
    blocked = np.asarray(record["blocked_controls"])
    controls = np.repeat(blocked, lengths, axis=1)
    close(controls, record["controls"])
    state = np.array(record["states"], dtype=float)
    history = []
    dt = config["dt"]
    for control in controls.transpose(1, 0, 2):
        state[:, :3] += dt*state[:, 3:]+.5*dt**2*control
        state[:, 3:] += dt*control
        history.append(state.copy())
    predicted = np.stack(history, axis=1)
    close(predicted, record["prediction"])
    n, horizon = predicted.shape[:2]
    pursuers = config["num_pursuers"]
    targets = np.asarray(config["targets"])
    position = predicted[:, :, :3]
    weights = dict(goal=2., evasion=50., pursuit=10., repulsion=2., obstacle=5., effort=.1)
    weights.update(config.get("adversarial_weights", {}))
    goal_weight = weights["goal"] if pursuers else 1.
    effort_weight = weights["effort"] if pursuers else .1
    terms = {key: np.zeros(n) for key in ("tracking", "effort", "repulsion", "evasion", "velocity", "terminal", "obstacle")}
    terms["effort"] = effort_weight*(controls**2).sum(axis=(1, 2))
    a = np.eye(6)
    a[:3, 3:] = dt*np.eye(3)
    b = np.vstack([.5*dt**2*np.eye(3), dt*np.eye(3)])
    q = np.diag([goal_weight]*3+[config["velocity_weight"]]*3)
    extra = solve_discrete_are(a, b, q, effort_weight*np.eye(3))-q
    close(extra, config["terminal_extra_matrix"])
    pair_squared = {}
    for i, j in combinations(range(n), 2):
        squared = ((position[i]-position[j])**2).sum(axis=1)
        pair_squared[i, j] = squared
        terms["repulsion"][[i, j]] += (weights["repulsion"] if pursuers else 1.)*np.sum(1/(squared+.01))
    for i in range(n):
        if i < pursuers:
            squared = ((position[i]-position[pursuers:])**2).sum(axis=2)
            terms["tracking"][i] = weights["pursuit"]*np.min(squared, axis=0).sum()
        else:
            target = targets[i-pursuers]
            terms["tracking"][i] = goal_weight*((position[i]-target)**2).sum()
            terms["velocity"][i] = config["velocity_weight"]*(predicted[i, :, 3:]**2).sum()
            error = predicted[i, -1]-np.r_[target, np.zeros(3)]
            terms["terminal"][i] = config["terminal_scale"]*error@extra@error
            if pursuers:
                squared = ((position[i]-position[:pursuers])**2).sum(axis=2)
                terms["evasion"][i] = np.sum(weights["evasion"]/(np.min(squared, axis=0)+.1))
        obstacles = np.asarray(config["obstacles"])
        if len(obstacles):
            squared = ((position[i, :, None]-obstacles[None])**2).sum(axis=2)
            terms["obstacle"][i] = (weights["obstacle"] if pursuers else 5.)*np.sum(1/(squared+.01))
    for key, value in terms.items():
        close(value, record["cost_terms"][key])
    close(sum(terms.values()), record["costs"])
    altitude_bounds = record.get("constraint_altitude_bounds", config["planned_altitude_bounds"])
    violations = [altitude_bounds[0]-position[:, :, 2],
                  position[:, :, 2]-altitude_bounds[1]]
    if config["hard_pair_separation"]:
        violations.extend(config["planned_separation_m"]**2-d for d in pair_squared.values())
    speed_bound = record.get("constraint_speed_bound", config.get("speed_bound"))
    jerk_bound = record.get("constraint_jerk_bound", config.get("jerk_bound"))
    if speed_bound is not None:
        violations.append(np.sum(predicted[:, :, 3:]**2, axis=2)-speed_bound**2)
    if jerk_bound is not None:
        previous = np.asarray(record["previous_controls"])[:, None]
        change = np.diff(np.concatenate([previous, blocked], axis=1), axis=1)
        violations.extend([change-dt*jerk_bound, -change-dt*jerk_bound])
    worst = max(float(v.max()) for v in violations)
    feasible = bool(np.isfinite(blocked).all() and np.abs(blocked).max() <= config.get("input_bound", 5.)+1e-8 and worst <= 1e-7)
    assert feasible == record["feasible"]
    assert controls.shape == (n, horizon, 3)
    return predicted, feasible, worst


def goal_summary(data, config, summary):
    indices = np.arange(config["num_pursuers"], config["num_agents"])
    errors = np.linalg.norm(data["states"][:, indices, :3]-data["targets"][None], axis=2)
    speed = np.linalg.norm(data["states"][:, indices, 3:], axis=2)
    close(errors[0], summary["initial_goal_distance_m"])
    close(errors[-1], summary["final_goal_distance_m"])
    close(speed[-1], summary["final_goal_speed_mps"])
    close(data["times"][-1], summary["simulated_duration_s"])
    assert np.isfinite(data["states"]).all()
    return errors, speed


def interval_extrema(states, acceleration, dt):
    """Independent polynomial construction for the first applied interval."""
    from numpy.polynomial import Polynomial
    minimum_distance = float("inf")
    for i, j in combinations(range(len(states)), 2):
        delta = states[i]-states[j]
        change = acceleration[i]-acceleration[j]
        squared = sum((Polynomial([delta[k], delta[k+3], .5*change[k]])**2
                       for k in range(3)), Polynomial([0.]))
        roots = squared.deriv().roots()
        times = [0., dt, *(float(r.real) for r in roots if abs(r.imag) < 1e-8 and 0 < r.real < dt)]
        minimum_distance = min(minimum_distance, float(np.sqrt(max(0., min(squared(times))))))
    heights = []
    for state, control in zip(states, acceleration):
        polynomial = Polynomial([state[2], state[5], .5*control[2]])
        times = [0., dt, *(float(r.real) for r in polynomial.deriv().roots()
                          if abs(r.imag) < 1e-8 and 0 < r.real < dt)]
        heights.extend(polynomial(times))
    maximum_speed = max(np.linalg.norm(states[:, 3:], axis=1).max(),
                        np.linalg.norm(states[:, 3:]+dt*acceleration, axis=1).max())
    return dict(first_interval_minimum_distance_m=minimum_distance,
                first_interval_minimum_altitude_m=float(min(heights)),
                first_interval_maximum_altitude_m=float(max(heights)),
                first_interval_maximum_speed_mps=float(maximum_speed))


def pid_telemetry(data, config, summary):
    if "requested_forces" not in data:
        return
    if len(data["times"]) == 1:
        for key in ("motor_saturation_fraction", "reference_position_rmse_m", "reference_velocity_rmse_mps"):
            assert summary[key] is None
        return
    integral = np.zeros((config["num_agents"], 3))
    for row in range(1, len(data["times"])):
        position_error = data["references"][row, :, :3]-data["states"][row-1, :, :3]
        velocity_error = data["references"][row, :, 3:]-data["states"][row-1, :, 3:]
        integral = np.clip(integral+position_error/120, -2., 2.)
        integral[:, 2] = np.clip(integral[:, 2], -.15, .15)
        force = position_error*[.4, .4, 1.25]+integral*.05+velocity_error*[.2, .2, .5]+[0., 0., .027*9.8]
        if config["pid_mode"] != "legacy":
            force += .027*data["accelerations"][row]
        close(force, data["requested_forces"][row])
        if config["pid_mode"] == "limited":
            force[:, 2] = np.maximum(force[:, 2], .2*.027*9.8)
            horizontal = np.linalg.norm(force[:, :2], axis=1)
            limit = force[:, 2]*np.tan(np.radians(25.))
            factor = np.ones(len(force))
            over = horizontal > limit
            factor[over] = limit[over]/horizontal[over]
            force[:, :2] *= factor[:, None]
            maximum = .8*4*3.16e-10*(.2685*65535+4070.3)**2
            force *= np.minimum(1., maximum/np.linalg.norm(force, axis=1))[:, None]
        close(force, data["commanded_forces"][row])
    expected_saturation = (np.isclose(data["rpms"][1:], .2685*20000+4070.3, atol=1e-6)
                           | np.isclose(data["rpms"][1:], .2685*65535+4070.3, atol=1e-6))
    # The implementation uses NumPy's default relative tolerance for RPM flags.
    np.testing.assert_array_equal(expected_saturation, data["motor_saturation"][1:])
    close(expected_saturation.mean(), summary["motor_saturation_fraction"])
    close(np.sqrt(np.mean(np.sum((data["states"][1:, :, :3]-data["references"][1:, :, :3])**2, axis=2))),
          summary["reference_position_rmse_m"])
    close(np.sqrt(np.mean(np.sum((data["states"][1:, :, 3:]-data["references"][1:, :, 3:])**2, axis=2))),
          summary["reference_velocity_rmse_mps"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate without changing archived reports or audit sources")
    args = parser.parse_args()
    from tools.reproduce import writable_output
    if args.simulation is not None and not getattr(args, "check_only", False): writable_output(args.simulation)
    if args.benchmark is not None and not getattr(args, "check_only", False): writable_output(args.benchmark)
    if args.benchmark is None and args.simulation is None:
        parser.error("Provide a simulation or benchmark archive")
    datasets = []
    roots = []
    if args.benchmark:
        sources(args.benchmark)
        nominal = read(args.benchmark/"nominal_summary.json")
        assert nominal["complete"]
        roots.append(args.benchmark)
        datasets.append((True, args.benchmark, nominal["cases"]))
    if args.simulation:
        sources(args.simulation)
        simulation = read(args.simulation/"summary.json")
        assert simulation["complete"]
        datasets.append((False, args.simulation, simulation["cases"]))
        roots.append(args.simulation)
    results = []
    for is_nominal, root, cases in datasets:
        for summary in cases:
            label = f"{summary['scenario']}_{summary['variant']}"
            directory = root/(f"nominal_{label}" if is_nominal else label)
            config_path = directory/"config.json"
            if not config_path.exists():
                assert args.simulation, "Older nominal archives need the matching simulation config"
                config_path = args.simulation/label/"config.json"
            config = read(config_path)
            with np.load(directory/"trajectory.npz") as archive:
                data = dict(archive)
            close(data["states"][0], config["initial_states"])
            close(data["targets"], config["targets"])
            errors, speed = goal_summary(data, config, summary)
            step = .1 if is_nominal else 1/120
            close(data["times"], np.arange(len(data["times"]))*step)
            reached = (errors <= .25)&(speed <= .3)
            recorded_reached = summary["reached_with_low_speed" if is_nominal else "ever_reached_with_low_speed"]
            np.testing.assert_array_equal(reached.any(axis=0), recorded_reached)
            applied, attempts, maximum_applied_violation = 0, 0, -np.inf
            previous_applied = np.zeros((config["num_agents"], 3))
            for path in sorted((directory/"attempts").glob("k*.json")):
                record = read(path)
                index = record["update"]
                if "previous_controls" in record:
                    close(previous_applied, record["previous_controls"])
                close(record["states"], data["states"][index if is_nominal else index*12])
                predicted, feasible, worst = evaluate(record, config)
                used = feasible if is_nominal else record["applied"]
                if used:
                    assert feasible
                    applied += 1
                    previous_applied = np.asarray(record["controls"])[:, 0]
                    maximum_applied_violation = max(maximum_applied_violation, worst)
                    if is_nominal:
                        close(data["states"][index+1], predicted[:, 0])
                    else:
                        extent = interval_extrema(np.asarray(record["states"]),
                                                  np.asarray(record["controls"])[:, 0], config["dt"])
                        for key, value in record["first_interval"].items():
                            close(extent[key], value)
                        assert extent["first_interval_minimum_altitude_m"] >= .3-1e-8
                        assert extent["first_interval_maximum_altitude_m"] <= 2.+1e-8
                        if config["hard_pair_separation"]:
                            assert extent["first_interval_minimum_distance_m"] >= .5-1e-8
                        if config.get("physical_speed_bound") is not None:
                            assert extent["first_interval_maximum_speed_mps"] <= config["physical_speed_bound"]+1e-8
                        initial = np.asarray(record["states"])
                        for tick in range(1, 13):
                            row = index*12+tick
                            if row >= len(data["times"]):
                                break
                            elapsed = tick/120
                            reference = initial.copy()
                            control = np.asarray(record["controls"])[:, 0]
                            reference[:, :3] += elapsed*initial[:, 3:]+.5*elapsed**2*control
                            reference[:, 3:] += elapsed*control
                            close(reference, data["references"][row])
                attempts += 1
            if not is_nominal:
                assert applied == summary["applied_updates"]
                close(np.linalg.norm(data["quaternions"], axis=2), 1.)
                distances = [np.linalg.norm(data["states"][:, i, :3]-data["states"][:, j, :3], axis=1)
                             for i, j in combinations(range(config["num_agents"]), 2)]
                close(np.min(distances), summary["minimum_measured_pair_distance_m"])
                close(data["states"][:, :, 2].min(), summary["minimum_measured_altitude_m"])
                close(data["states"][:, :, 2].max(), summary["maximum_measured_altitude_m"])
                pid_telemetry(data, config, summary)
            else:
                assert applied == len(data["times"])-1
            results.append(dict(case=label, plant="nominal" if is_nominal else "pybullet",
                                attempts=attempts, applied=applied,
                                maximum_applied_constraint_violation=maximum_applied_violation, passed=True))
    report = dict(passed=True, cases=results,
                  scope="Independent dynamics, costs, constraints and recorded-flight consistency.")
    for root in ([] if args.check_only else roots):
        (root/"independent_validation.json").write_text(json.dumps(report, indent=2)+"\n")
        dest = root/"audit_sources"/Path(__file__).name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=True, cases=len(results), attempts=sum(r["attempts"] for r in results))))


if __name__ == "__main__":
    main()
