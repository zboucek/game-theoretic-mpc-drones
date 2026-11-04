# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Controlled-study execution and record association for article IV-A.

Study runs fixed starts and receding-horizon episodes, then associates distinct
accepted candidates with unilateral searches. Archive owns record I/O;
tools/results.py recomputes the reported counts and table."""
import time

import numpy as np

from .archive import write_json
from .model import NumpyModel, step
from .protocol import (GEOMETRIES, INITIALIZATIONS, LOOP_STARTS, METHODS, PROTOCOL,
                       initial_controls, minimum_distance, plan, shift_controls)
from .solvers import Solvers, deviation_attempt, deviation_jobs


class CheckpointStop(Exception):
    pass


def identity(case, start, method):
    return f"{case['geometry']}-p{case['perturbation']}-{start}-{method}"


def progress_metrics(states, initial_states, targets):
    initial = np.asarray(initial_states)
    states, targets = np.asarray(states), np.asarray(targets)
    distance = np.linalg.norm(states[:, :3]-targets[:, :3], axis=1)
    original_distance = np.linalg.norm(initial[:, :3]-targets[:, :3], axis=1)
    moving = original_distance > .25
    direction = targets[:, :2]-initial[:, :2]
    norm = np.linalg.norm(direction, axis=1)
    direction = direction/np.maximum(norm[:, None], 1e-15)
    # All scenarios cross the horizontal origin. Holding agents are excluded.
    passed = np.sum(states[:, :2]*direction, axis=1) >= .25
    return {"goal_distance_m": distance, "progress_m": original_distance-distance,
            "conflict_passed": bool(np.all(passed[moving])),
            "per_player_conflict_passed": passed, "moving_players": moving,
            "all_targets_reached": bool(np.all(distance <= .25))}


def candidate_groups(records, config=PROTOCOL):
    groups = []
    for record in sorted(records, key=lambda r: r["record_id"]):
        if not record["validation"]["accepted"]:
            continue
        equivalent = next((g for g in groups if
            g["representative"]["geometry"] == record["geometry"] and
            g["representative"]["perturbation"] == record["perturbation"] and
            np.max(np.abs(np.asarray(g["representative"]["controls"])-record["controls"]))
                <= config.unique_control_tolerance), None)
        if equivalent is None:
            groups.append({"representative": record, "members": [record["record_id"]]})
        else:
            equivalent["members"].append(record["record_id"])
    return groups


class Study:
    def __init__(self, archive, max_new_attempts=None, solvers=None):
        self.archive = archive
        self.config = archive.config
        self.solvers = solvers or Solvers(self.config)
        self.limit = max_new_attempts
        self.new_attempts = 0

    def attempt(self, key, compute):
        previous = self.archive.get(key)
        if previous is not None:
            return previous
        if self.limit is not None and self.new_attempts >= self.limit:
            raise CheckpointStop("Requested attempt budget reached; checkpoint saved")
        event = self.archive.event("attempt_started", record_id=key)
        try:
            result = compute()
        except Exception as error:
            self.archive.event("attempt_exception", record_id=key, start_event=event,
                               exception=type(error).__name__, message=str(error))
            raise
        result["start_event"] = event
        saved = self.archive.save(key, result)
        self.archive.event("attempt_finished", record_id=key, start_event=event)
        self.new_attempts += 1
        print(f"{key}: " + (f"accepted={result['validation']['accepted']} "
              f"solve={result['solver']['solve_seconds']:.4g}s" if "validation" in result
              else f"refutes_gne={result.get('refutes_gne')}"), flush=True)
        return saved

    def prepare(self, phase):
        checks = {}
        for case in plan(phase)["instances"]:
            if case["perturbation"]:
                continue
            states, targets = np.asarray(case["states"]), np.asarray(case["targets"])
            for method in METHODS:
                record = self.solvers.prepare(method, states, targets)
                if record:
                    self.archive.event("preparation", **record)
            checks[case["geometry"]] = self.solvers.derivative_check(states, targets)
            checks[case["geometry"]]["parameter_reuse"] = {
                method: self.solvers.parameter_reuse_check(method, states, targets) for method in METHODS}
            checks[case["geometry"]]["passed"] &= all(
                c["passed"] for c in checks[case["geometry"]]["parameter_reuse"].values())
        write_json(self.archive.directory / "implementation_checks.json",
                   {"checks": checks, "passed": all(c["passed"] for c in checks.values()),
                    "preparations": self.solvers.preparations})
        if not all(c["passed"] for c in checks.values()):
            raise RuntimeError("Independent derivative check failed; measurement withheld")

    def statics(self, phase):
        for case in plan(phase)["instances"]:
            states, targets = np.asarray(case["states"]), np.asarray(case["targets"])
            for start in INITIALIZATIONS:
                guess = initial_controls(states, case["geometry"], start, self.config)
                for method in METHODS:
                    key = "static-"+identity(case, start, method)
                    self.attempt(key, lambda: dict(self.solvers.solve(method, states, targets, guess),
                        kind="static", geometry=case["geometry"], perturbation=case["perturbation"],
                        initialization=start))

    def loops(self, phase):
        design = plan(phase)
        for case in design["instances"]:
            for start in LOOP_STARTS:
                for method in METHODS:
                    episode = identity(case, start, method)
                    states = np.asarray(case["states"])
                    targets = np.asarray(case["targets"])
                    guess = initial_controls(states, case["geometry"], start, self.config)
                    applied = [states]
                    records = []
                    reason = "update_limit"
                    for k in range(design["updates_per_episode"]):
                        key = f"loop-{episode}-k{k:03d}"
                        record = self.attempt(key, lambda: dict(
                            self.solvers.solve(method, states, targets, guess), kind="loop",
                            geometry=case["geometry"], perturbation=case["perturbation"],
                            initialization=start, episode=episode, update=k, simulated_time_s=k*self.config.dt))
                        # Resume must follow this method's own accepted sequence exactly.
                        if not np.allclose(record["states"], states, atol=1e-13, rtol=0):
                            raise ValueError(f"Checkpoint state mismatch in {episode}")
                        if not np.allclose(record["initial_controls"], guess, atol=1e-13, rtol=0):
                            raise ValueError(f"Checkpoint warm start mismatch in {episode}")
                        records.append(record)
                        if not record["validation"]["accepted"]:
                            reason = "numerical_failure"
                            break
                        controls = np.asarray(record["controls"]).reshape(3, self.config.horizon, 3)
                        states = step(states, controls[:, 0], self.config)
                        np.testing.assert_allclose(states, np.asarray(record["prediction"])[:, 0], atol=1e-11, rtol=0)
                        applied.append(states)
                        guess = shift_controls(controls, self.config)
                    accepted = [r for r in records if r["validation"]["accepted"]]
                    metrics = progress_metrics(states, case["states"], targets)
                    history = [progress_metrics(x, case["states"], targets) for x in applied]
                    summary = {"kind": "episode", "phase": phase, "episode": episode,
                        "geometry": case["geometry"], "perturbation": case["perturbation"],
                        "initialization": start, "method": method, "requested_updates": design["updates_per_episode"],
                        "accepted_updates": len(accepted), "termination": reason,
                        "attempt_ids": [r["record_id"] for r in records],
                        "applied_states": np.asarray(applied),
                        "simulated_time_s": np.arange(len(applied))*self.config.dt,
                        "minimum_executed_node_distance_m": min(minimum_distance(x) for x in applied),
                        "minimum_predicted_distance_m": min((r["validation"]["minimum_predicted_distance_m"]
                                                               for r in accepted), default=None),
                        "cumulative_player_costs": np.sum([np.asarray(r["stage_costs"])[:, 0] for r in accepted], axis=0)
                            if accepted else np.zeros(3),
                        "cumulative_player_effort": np.sum([np.sum(np.asarray(r["controls"]).reshape(3, -1, 3)[:, 0]**2,
                            axis=1) for r in accepted], axis=0) if accepted else np.zeros(3),
                        "first_accepted": accepted[0]["record_id"] if accepted else None,
                        "last_accepted": accepted[-1]["record_id"] if accepted else None,
                        "conflict_ever_passed": any(m["conflict_passed"] for m in history),
                        "first_conflict_passage_s": next((k*self.config.dt for k, m in enumerate(history)
                                                         if m["conflict_passed"]), None), **metrics}
                    # Episode summaries are derived; raw attempt files are never replaced.
                    write_json(self.archive.directory / "episodes" / phase / f"{episode}.json", summary)

    def diagnose(self, phase):
        max_perturbation = 0 if phase == "pilot" else 4
        groups = candidate_groups([r for r in self.archive.records("static") if r["perturbation"] <= max_perturbation])
        associations = {}
        representatives = {}
        for group in groups:
            candidate = group["representative"]
            for member in group["members"]:
                associations[member] = candidate["record_id"]
            representatives[candidate["record_id"]] = candidate
        for path in sorted((self.archive.directory / "episodes" / phase).glob("*.json")):
            import json
            episode = json.loads(path.read_text())
            for role in ("first_accepted", "last_accepted"):
                key = episode[role]
                if key:
                    representatives[key] = self.archive.get(key)
                    associations[key] = key
        write_json(self.archive.directory / f"diagnostic_groups_{phase}.json", associations)
        for candidate_id, record in sorted(representatives.items()):
            for player, start, guess in deviation_jobs(record, record["geometry"], self.config):
                key = f"deviation-{candidate_id}-i{player}-{start}"
                self.attempt(key, lambda: dict(deviation_attempt(record, player, start, guess, self.config),
                    kind="deviation", candidate_id=candidate_id,
                    geometry=record["geometry"], perturbation=record["perturbation"]))

    def run(self, phase):
        started = time.perf_counter()
        self.prepare(phase)
        self.statics(phase)
        self.loops(phase)
        self.diagnose(phase)
        self.archive.event("phase_finished", phase=phase, wall_seconds=time.perf_counter()-started)
