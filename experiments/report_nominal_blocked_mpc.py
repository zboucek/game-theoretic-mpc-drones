#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Plot goal arrival, dwell and pursuit distance from saved nominal MPC data."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-theory-mpc-nominal-report")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def first_time(times, condition):
    indices = np.flatnonzero(condition)
    return float(times[indices[0]]) if len(indices) else None


def longest_dwell(times, condition):
    edges = np.diff(np.r_[False, condition, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1
    return float(max((times[b]-times[a] for a, b in zip(starts, ends)), default=0.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    from tools.reproduce import writable_output
    if args.source is not None and not getattr(args, "check_only", False): writable_output(args.source)
    summaries = read(args.source/"nominal_summary.json")
    assert summaries["complete"], "Wait for the numerical run to finish"
    cases = summaries["cases"]
    fig, axes = plt.subplots(3, len(cases), figsize=(4.4*len(cases), 8.8),
                             squeeze=False, layout="constrained")
    records, inputs, response_stats = [], [], []
    colors = ["#1768ac", "#d97706", "#24845b"]
    for column, summary in enumerate(cases):
        directory = args.source/f"nominal_{summary['scenario']}_{summary['variant']}"
        config = read(directory/"config.json")
        attempts = [read(path) for path in sorted((directory/"attempts").glob("k*.json"))]
        changes = [r["response_change"] for r in attempts if r["response_change"] is not None]
        if changes:
            response_stats.append(dict(variant=summary["variant"], updates=len(changes),
                last_response_change=float(changes[-1]), median_response_change=float(np.median(changes)),
                sweep_changes_below_1e_minus_3=sum(c < 1e-3 for c in changes)))
        with np.load(directory/"trajectory.npz") as data:
            states, times, targets = data["states"], data["times"], data["targets"]
        indices = np.arange(config["num_pursuers"], config["num_agents"])
        distance = np.linalg.norm(states[:, indices, :3]-targets[None], axis=2)
        speed = np.linalg.norm(states[:, indices, 3:], axis=2)
        n_pursuers = config["num_pursuers"]
        nearest = (np.linalg.norm(states[:, indices, None, :3]-states[:, None, :n_pursuers, :3], axis=3).min(axis=2)
                   if n_pursuers else None)
        for local in range(len(indices)):
            reached = (distance[:, local] <= .25)&(speed[:, local] <= .3)
            before_end = times >= max(0., times[-1]-10.)
            label = f"Evader {local+1}" if n_pursuers else f"Agent {local+1}"
            first_goal = first_time(times, reached)
            capture = first_time(times, nearest[:, local] < .3) if n_pursuers else None
            records.append(dict(scenario=summary["scenario"], variant=summary["variant"], agent=label,
                duration_s=float(times[-1]), first_goal_stop_time_s=first_goal,
                longest_goal_dwell_s=longest_dwell(times, reached),
                fraction_last_10s_at_goal=float(reached[before_end].mean()),
                final_goal_distance_m=float(distance[-1, local]), final_speed_mps=float(speed[-1, local]),
                minimum_goal_distance_m=float(distance[:, local].min()),
                first_capture_threshold_time_s=capture,
                minimum_pursuer_distance_m=float(nearest[:, local].min()) if n_pursuers else None))
            axes[0, column].plot(times, distance[:, local], label=label, color=colors[local])
            axes[1, column].plot(times, speed[:, local], color=colors[local])
            if nearest is not None:
                axes[2, column].plot(times, nearest[:, local], color=colors[local])
            if first_goal is not None:
                index = np.searchsorted(times, first_goal)
                axes[0, column].plot(first_goal, distance[index, local], "o", color=colors[local], markersize=5)
            if capture is not None:
                index = np.searchsorted(times, capture)
                axes[2, column].plot(capture, nearest[index, local], "x", color=colors[local], markersize=7)
        axes[0, column].set_title(f"{summary['variant']}: H={config['horizon']}, B={config['control_blocks']}")
        axes[0, column].axhline(.25, color="gray", linestyle=":", linewidth=.8)
        axes[1, column].axhline(.3, color="gray", linestyle=":", linewidth=.8)
        axes[2, column].axhline(.3, color="gray", linestyle=":", linewidth=.8)
        axes[0, column].legend(fontsize=8)
        for row, ylabel in enumerate(("Assigned-goal distance [m]", "Speed [m/s]", "Nearest pursuer distance [m]")):
            axes[row, column].set(xlabel="Simulated time [s]", ylabel=ylabel)
            axes[row, column].grid(alpha=.25)
        inputs.extend(directory/name for name in ("config.json", "trajectory.npz", "summary.json"))
    fig.suptitle("Nominal double integrator · dots mark first goal arrival at low speed", fontsize=12)
    fig.supxlabel("Goal: distance ≤ 0.25 m and speed ≤ 0.30 m/s. × marks capture-threshold crossing.\n"
                  "No absorbing capture/goal policy; pursuit continues.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.source/f"nominal_goal_dwell.{extension}", dpi=170)
    plt.close(fig)
    (args.source/"nominal_events.json").write_text(json.dumps(dict(cases=records, response_policy_stats=response_stats,
        goal_definition="distance <= .25 m AND speed <= .3 m/s",
        dwell_definition="Time between first and last samples in the longest uninterrupted run satisfying both goal conditions",
        capture_definition="First nearest-pursuer distance < .3 m; dynamics continue, no absorbing event"), indent=2)+"\n")
    def show(value):
        return "—" if value is None else f"{value:.2f}"
    lines = ["", "<!-- nominal-diagnostics -->", "", "## Arrival and dwell in the longer nominal diagnostic", "",
        "These trajectories use the nominal double integrator, not PyBullet. Pursuit continues after a goal or capture-threshold crossing; no absorbing terminal policy is imposed.", "",
        "Arrival requires goal distance ≤ 0.25 m **and** speed ≤ 0.30 m/s. Dwell is the longest uninterrupted interval satisfying both.", "",
        "| Variant | Agent | First goal stop [s] | Longest goal dwell [s] | Final distance [m] | Final speed [m/s] | First capture threshold [s] |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['variant']} | {row['agent']} | {show(row['first_goal_stop_time_s'])} | "
                     f"{row['longest_goal_dwell_s']:.2f} | {row['final_goal_distance_m']:.3f} | "
                     f"{row['final_speed_mps']:.3f} | {show(row['first_capture_threshold_time_s'])} |")
    lines += ["", "![Goal distances, speeds and pursuer distances](nominal_goal_dwell.png)", ""]
    if response_stats:
        lines += ["The controller permits only three damped response sweeps. A small last-sweep control change is not a Nash certificate; large changes indicate an unsettled response iteration.", "",
                  "| Variant | Median last-sweep input change [m/s²] | Updates with change < 0.001 |",
                  "|---|---:|---:|"]
        for row in response_stats:
            lines.append(f"| {row['variant']} | {row['median_response_change']:.3f} | {row['sweep_changes_below_1e_minus_3']}/{row['updates']} |")
        lines += ["", "Acceleration and altitude are bounded; this nominal diagnostic has no hard speed bound and is not a physical flight validation.", ""]
    report = args.source/"report.md"
    existing = report.read_text().split("\n<!-- nominal-diagnostics -->")[0]
    if not read(args.source/"timing.json")["samples"]:
        table = existing[existing.index("| Scenario | Variant | Duration"):]
        existing = ("# Longer nominal MPC simulation\n\n"
                    "Nominal double-integrator plant; this run measures trajectories, not solver scalability.\n"
                    "The horizon and solver policies match the earlier 10-second diagnostic.\n\n"+table)
    report.write_text(existing.rstrip()+"\n"+"\n".join(lines))
    snapshot = args.source/"plot_sources"/Path(__file__).name
    snapshot.parent.mkdir(exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    (args.source/"plot_manifest.json").write_text(json.dumps(dict(
        renderer_source=str(snapshot.relative_to(args.source)), renderer_sha256=sha256(snapshot.read_bytes()).hexdigest(),
        inputs={str(path.relative_to(args.source)): sha256(path.read_bytes()).hexdigest() for path in inputs},
        artifacts={name: sha256((args.source/name).read_bytes()).hexdigest()
                   for name in ("nominal_goal_dwell.png", "nominal_goal_dwell.pdf", "nominal_events.json")}), indent=2)+"\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
