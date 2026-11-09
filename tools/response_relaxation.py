# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Independently check the retained PyBullet response-relaxation comparison."""
from collections import Counter
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def analyze(directory, logs):
    from tools.results import module

    configs, result = [], {}
    for arm, fractions in (("damped", [0.7, 0.35, 1.0]), ("full", [1.0])):
        root = directory / arm
        module("audit_blocked_terminal_mpc", ["--simulation", root, "--check-only"],
               logs / f"relaxation-{arm}.log")
        flight = root / "adversarial_terminal"
        config = read(flight / "config.json")
        if config.pop("response_fractions") != fractions:
            raise ValueError(f"Unexpected response fractions: {arm}")
        configs.append(config)
        with np.load(flight / "trajectory.npz") as data:
            times, states, targets = data["times"], data["states"], data["targets"]
        error = np.linalg.norm(states[:, 3:, :3] - targets[None], axis=2)
        nearest = np.linalg.norm(states[:, 3:, None, :3] - states[:, None, :3, :3], axis=3).min()
        if nearest < 0.30:
            raise ValueError(f"Capture threshold crossed: {arm}")
        arrived = ((error <= 0.25) & (np.linalg.norm(states[:, 3:, 3:], axis=2) <= 0.3)).all(axis=1)
        counts = Counter()
        paths = sorted((flight / "attempts").glob("k*.json"))
        for path in paths:
            record = read(path)
            calls = [record] + ([record["tightened_attempt"]] if "tightened_attempt" in record else [])
            for call in calls:
                if call["response_fractions"] != fractions:
                    raise ValueError(f"Response fraction mismatch: {path}")
                for attempt in call["attempts"]:
                    value = attempt["fraction"]
                    if value not in [0.0, *fractions]:
                        raise ValueError(f"Unexpected selected fraction: {path}")
                    counts[str(value)] += 1
        result[arm] = dict(duration_s=float(times[-1]),
                           joint_arrival_s=float(times[np.flatnonzero(arrived)[0]]) if arrived.any() else None,
                           selected_fractions=dict(counts), selected_35_percent=counts["0.35"])
        if arm == "full":
            record = read(paths[-1])
            if record["applied"] or record["feasible"]:
                raise ValueError("Expected an unapplied, infeasible final full-update plan")
            state = np.asarray(record["states"])
            dt = config["dt"]
            upper = state[:, 2] + dt * state[:, 5] + 0.5 * dt**2 * config["input_bound"]
            result[arm]["maximum_next_altitude_m"] = float(upper.min())
            result[arm]["lower_altitude_bound_m"] = config["physical_altitude_bounds"][0]
            if upper.min() >= config["physical_altitude_bounds"][0]:
                raise ValueError("Full-update rejection has no first-node altitude witness")
    if configs[0] != configs[1]:
        raise ValueError("Response-relaxation flight configurations are not matched")
    return result
