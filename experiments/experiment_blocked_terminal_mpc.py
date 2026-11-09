#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""PyBullet flight loop for cooperative and adversarial experiments (VI-A/B).

run_case connects measured states, planner solves, execution checks and PID.
Article profiles are selected by tools/reproduce.py; physics pauses for solves."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-theory-mpc-terminal-plots")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.equilibrium_selection.archive import write_json
from experiments.equilibrium_mpc_execution import first_interval_extrema
from game_theory_mpc.blocked_game_mpc import AdversarialWeights, BlockedGameMPC, CooperativeWeights


def scenario(name, cooperative_layout="head-on"):
    if name == "cooperative":
        if cooperative_layout not in ("head-on", "all-moving"):
            raise ValueError("Unknown cooperative layout")
        states = np.array([[-.3, 0, 1, .2, 0, 0], [.3, 0, 1, -.1, 0, 0], [0, 2, 1, 0, 0, 0]])
        targets = np.array([[10., 0, 1], [-6., 0, 1], [0, 2, 1]])
        if cooperative_layout == "all-moving":
            targets[2] = [0., -2., 1.]
        return dict(states=states, targets=targets, obstacles=np.empty((0, 3)), pursuers=0, short_horizon=5)
    return dict(states=np.array([[2, 2, 1, 0, 0, 0], [2, -2, 1, 0, 0, 0], [-2, 2, 1, 0, 0, 0],
                                 [-1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=float),
                targets=np.array([[3., 1, 1], [-3., -1, 1]]),
                obstacles=np.array([[0, 1., 1], [0, -1., 1], [1.5, .5, 1], [-1.5, -.5, 1]]),
                pursuers=3, short_horizon=8)


def apply_preset(args, arguments):
    if args.preset != "adversarial-goal-tracking":
        return
    explicit = {value.split("=", 1)[0] for value in arguments if value.startswith("--")}
    values = dict(scenarios=["adversarial"], variants=["terminal"], updates=600,
        goal_weight=100., evasion_weight=10., velocity_weight=5., terminal_scale=1.,
        input_bound=2., speed_bound=2., physical_speed_bound=2., jerk_bound=6.,
        emergency_jerk_bound=6., response_sweeps=5, pid_mode="legacy", reserve_retry=True)
    for name, value in values.items():
        if "--"+name.replace("_", "-") not in explicit:
            setattr(args, name, value)


def run_case(name, variant, args, output, case_override=None):
    import pybullet as bullet
    from game_theory_mpc.acceleration_pid import AccelerationPIDControl
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    case = scenario(name, args.cooperative_layout) if case_override is None else case_override
    states, targets = case["states"], case["targets"]
    n, pursuers = len(states), case["pursuers"]
    lengths = (1,)*case["short_horizon"] if variant == "baseline" else tuple(args.blocks)
    weights = getattr(args, "adversarial_weights", None) or AdversarialWeights(goal=args.goal_weight, evasion=args.evasion_weight)
    cooperative_weights = CooperativeWeights(position=getattr(args, "position_weight", 1.),
        repulsion=getattr(args, "repulsion_weight", 1.), effort=getattr(args, "effort_weight", .1))
    physical_speed_bound = args.physical_speed_bound
    if physical_speed_bound is None and args.speed_bound is not None:
        physical_speed_bound = args.speed_bound+.5
    model_type = BlockedGameMPC
    backend = "primal"
    if args.equilibrium_method != "potential":
        if name != "cooperative":
            raise ValueError("GNE/vGNE selection is for cooperative flight")
        from functools import partial
        from game_theory_mpc.cooperative_equilibrium_mpc import CooperativeEquilibriumMPC
        model_type = partial(CooperativeEquilibriumMPC, method=args.equilibrium_method, max_sweeps=getattr(args, "max_sweeps", 30))
    dt = getattr(args, "dt", .1)
    pid_hz, physics_hz = getattr(args, "pid_hz", 120), getattr(args, "physics_hz", 240)
    physical_altitude = getattr(args, "physical_altitude_bounds", (.3, 2.))
    physical_separation = getattr(args, "physical_separation", .5)
    shared_options = dict(dt=dt, separation=getattr(args, "separation", .55))
    model = model_type(n, lengths, altitude_bounds=getattr(args, "altitude_bounds", (.35, 1.95)), **shared_options, num_pursuers=pursuers, obstacles=case["obstacles"],
                           velocity_weight=args.velocity_weight,
                           terminal_scale=args.terminal_scale if variant == "terminal" else 0.,
                           max_iterations=args.max_iterations, response_sweeps=args.response_sweeps,
                           response_fractions=getattr(args, "response_fractions", (.7, .35, 1.)),
                           adversarial_weights=weights, input_bound=args.input_bound,
                           speed_bound=args.speed_bound, jerk_bound=args.jerk_bound,
                           cooperative_weights=cooperative_weights)
    directory = output / f"{name}_{variant}"
    directory.mkdir()
    config = dict(scenario=name, variant=variant, initial_states=states, targets=targets,
                  cooperative_layout=args.cooperative_layout if name == "cooperative" else None,
                  obstacles=case["obstacles"], num_agents=n, num_pursuers=pursuers,
                  dt=dt, physics_hz=physics_hz, pid_hz=pid_hz, updates=args.updates,
                  horizon=model.horizon, block_lengths=lengths, control_blocks=model.blocking.blocks,
                  primal_variables=model.nvar, shared_inequalities=model.ng,
                  velocity_weight=model.velocity_weight, terminal_scale=model.terminal_scale,
                  response_fractions=model.response_fractions, response_sweeps=model.response_sweeps,
                  adversarial_weights=asdict(weights), input_bound=model.input_bound,
                  cooperative_weights=asdict(cooperative_weights),
                  speed_bound=model.speed_bound, jerk_bound=model.jerk_bound,
                  physical_speed_bound=physical_speed_bound, emergency_jerk_bound=args.emergency_jerk_bound,
                  pid_mode=args.pid_mode, maximum_pid_tilt_degrees=25. if args.pid_mode == "limited" else None,
                  equilibrium_method=args.equilibrium_method,
                  solver_backend=backend,
                  candidate_repair_enabled=getattr(model, "repair_deviations", False),
                  terminal_extra_matrix=model.P, terminal_convention="P_DARE minus the stage Q already present at the last predicted node",
                  stage_weights=dict(position=model.position_weight, input=model.effort_weight,
                                     pair_repulsion=weights.repulsion if pursuers else cooperative_weights.repulsion, evasion=weights.evasion if pursuers else 0.),
                  planned_altitude_bounds=model.altitude_bounds, physical_altitude_bounds=physical_altitude,
                  hard_pair_separation=model.hard_pairs, planned_separation_m=model.separation,
                  physical_separation_m=physical_separation if model.hard_pairs else None,
                  reserve_retry=args.reserve_retry,
                  notes="All cases start airborne at 1 m; this is a new matched comparison, not a replay of the old ground-start adversarial archive")
    write_json(directory / "config.json", config)
    prepared = model.prepare(states, targets)
    write_json(directory / "preparation.json", dict(seconds=prepared, details=getattr(model, "preparation", {})))
    pilot_event = getattr(args, "pilot_event", lambda **kwargs: None)
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=n, initial_xyzs=states[:, :3],
                    physics=Physics.PYB, pyb_freq=physics_hz, ctrl_freq=pid_hz, gui=False,
                    record=False, obstacles=False, user_debug_gui=False)
    history = {key: [] for key in ("times", "states", "quaternions", "references", "rpms",
        "requested_forces", "commanded_forces", "target_eulers", "motor_saturation", "force_limited", "accelerations")}
    attempts, capture_events = [], []
    physical_model = None
    termination = "update_limit"
    applied = 0
    started = perf_counter()
    try:
        env.reset()
        for drone, state in zip(env.DRONE_IDS, states):
            bullet.resetBaseVelocity(int(drone), linearVelocity=state[3:].tolist(), physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        observation = env._computeObs()
        pids = [AccelerationPIDControl(drone_model=DroneModel.CF2X,
            feedforward=args.pid_mode != "legacy", max_tilt_degrees=25. if args.pid_mode == "limited" else None) for _ in range(n)]
        history["times"].append(0.)
        history["states"].append(states.copy())
        history["quaternions"].append(observation[:, 3:7].copy())
        history["references"].append(states.copy())
        history["rpms"].append(np.zeros((n, 4)))
        for key in ("requested_forces", "commanded_forces", "target_eulers", "accelerations"):
            history[key].append(np.zeros((n, 3)))
        history["motor_saturation"].append(np.zeros((n, 4), dtype=bool))
        history["force_limited"].append(np.zeros(n, dtype=bool))
        if getattr(args, "checkpoint_updates", False):
            save_trajectory(directory, history, targets)
        for update in range(args.updates):
            pilot_event(stage="update", update=update, timeout_seconds=120.)
            update_started = perf_counter()
            measured = np.c_[observation[:, :3], observation[:, 10:13]]
            previous = model._previous_controls.copy()
            controls, prediction, info = model.solve(measured, targets)
            info["constraint_altitude_bounds"] = model.altitude_bounds
            info["constraint_speed_bound"] = model.speed_bound
            info["constraint_jerk_bound"] = model.jerk_bound
            if args.reserve_retry and not info["feasible"]:
                # A shifted plan need not satisfy the tightened reserve after
                # measured tracking error. Retry the same objective against
                # physical bounds; never bypass those bounds or the monitor.
                tightened_attempt = info.copy()
                if physical_model is None:
                    physical_model = model_type(n, lengths, **shared_options, num_pursuers=pursuers,
                        obstacles=case["obstacles"], velocity_weight=model.velocity_weight,
                        terminal_scale=model.terminal_scale, altitude_bounds=physical_altitude,
                        max_iterations=args.max_iterations, response_sweeps=args.response_sweeps,
                        response_fractions=getattr(args, "response_fractions", (.7, .35, 1.)),
                        adversarial_weights=weights, input_bound=args.input_bound,
                        speed_bound=min(args.speed_bound+getattr(args, "recovery_speed_margin", .2), physical_speed_bound) if args.speed_bound is not None else None,
                        jerk_bound=args.emergency_jerk_bound if args.jerk_bound is not None else None,
                        cooperative_weights=cooperative_weights)
                    retry_prepared = physical_model.prepare(measured, targets)
                else:
                    retry_prepared = 0.
                physical_model._warm = np.asarray(info["blocked_controls"]).copy()
                physical_model.commit_controls(previous)
                physical_model._solved = True
                controls, prediction, info = physical_model.solve(measured, targets)
                info["compute_time"] += tightened_attempt["compute_time"]
                info["retry_preparation_seconds"] = retry_prepared
                for timing in ("projection_seconds", "solver_seconds", "checks_seconds"):
                    if timing in info:
                        info[timing] += tightened_attempt.get(timing, 0.)
                info["constraint_altitude_bounds"] = physical_model.altitude_bounds
                info["constraint_speed_bound"] = physical_model.speed_bound
                info["constraint_jerk_bound"] = physical_model.jerk_bound
                info["tightened_attempt"] = tightened_attempt
                model._warm = physical_model._warm.copy()
            from game_theory_mpc.experiment_model import check_model
            checked_model = physical_model if "tightened_attempt" in info else model
            info["independent"] = check_model(checked_model, info["blocked_controls"], measured, targets)
            execution_checked_at = perf_counter()
            extent = first_interval_extrema(measured, controls[:, 0], dt)
            extent["first_interval_maximum_speed_mps"] = float(max(np.linalg.norm(measured[:, 3:], axis=1).max(),
                np.linalg.norm(measured[:, 3:]+dt*controls[:, 0], axis=1).max()))
            admissible = bool(info["feasible"] and extent["first_interval_minimum_altitude_m"] >= physical_altitude[0]-1e-8
                and extent["first_interval_maximum_altitude_m"] <= physical_altitude[1]+1e-8
                and (physical_speed_bound is None or extent["first_interval_maximum_speed_mps"] <= physical_speed_bound+1e-8)
                and (not model.hard_pairs or extent["first_interval_minimum_distance_m"] >= physical_separation-1e-8))
            record = dict(update=update, simulated_time_s=history["times"][-1], states=measured,
                          targets=targets, controls=controls, prediction=prediction, first_interval=extent,
                          previous_controls=previous,
                          execution_checks_seconds=perf_counter()-execution_checked_at,
                          external_wall_seconds=perf_counter()-update_started,
                          applied=admissible, execution_complete=False, **info)
            write_json(directory / "attempts" / f"k{update:03d}.json", record)
            attempts.append(record)
            print(f"{name}/{variant} {update+1}/{args.updates}: apply={admissible} "
                  f"solve={info['compute_time']:.3f}s", flush=True)
            if not admissible:
                termination = "no_feasible_control"
                break
            model.commit_controls(controls[:, 0])
            for tick in range(round(dt*pid_hz)):
                elapsed = (tick+1)/pid_hz
                reference = measured.copy()
                reference[:, :3] += elapsed*measured[:, 3:]+.5*elapsed**2*controls[:, 0]
                reference[:, 3:] += elapsed*controls[:, 0]
                rpms = np.stack([pid.computeControl(
                    control_timestep=1/pid_hz, cur_pos=observation[i, :3], cur_quat=observation[i, 3:7],
                    cur_vel=observation[i, 10:13], cur_ang_vel=observation[i, 13:16],
                    target_pos=reference[i, :3], target_vel=reference[i, 3:],
                    target_acceleration=controls[i, 0])[0]
                    for i, pid in enumerate(pids)])
                observation, _, terminated, truncated, _ = env.step(rpms)
                current = np.c_[observation[:, :3], observation[:, 10:13]]
                time = len(history["times"])/pid_hz
                history["times"].append(time)
                history["states"].append(current)
                history["quaternions"].append(observation[:, 3:7].copy())
                history["references"].append(reference)
                history["rpms"].append(rpms)
                for key, diagnostic in (("requested_forces", "requested_force_n"), ("commanded_forces", "commanded_force_n"),
                    ("target_eulers", "target_euler_rad"), ("motor_saturation", "motor_saturated"),
                    ("force_limited", "force_limited")):
                    history[key].append(np.asarray([pid.diagnostics[diagnostic] for pid in pids]))
                history["accelerations"].append(controls[:, 0].copy())
                contacts = [c for c in bullet.getContactPoints(physicsClientId=env.CLIENT)
                            if c[1] in env.DRONE_IDS or c[2] in env.DRONE_IDS]
                distance = min(np.linalg.norm(current[i, :3]-current[j, :3]) for i, j in combinations(range(n), 2))
                if not np.isfinite(current).all():
                    termination = "nonfinite_physics"
                elif contacts:
                    termination = "physical_contact"
                elif current[:, 2].min() < physical_altitude[0] or current[:, 2].max() > physical_altitude[1] or (model.hard_pairs and distance < physical_separation):
                    termination = "measured_limit_violation"
                elif physical_speed_bound is not None and np.linalg.norm(current[:, 3:], axis=1).max() > physical_speed_bound:
                    termination = "measured_speed_limit_violation"
                elif pursuers:
                    for evader in range(n-pursuers):
                        nearest = np.linalg.norm(current[:pursuers, :3]-current[pursuers+evader, :3], axis=1).min()
                        if nearest < .3:
                            capture_events.append(dict(evader=evader+1, time_s=time, minimum_distance_m=float(nearest)))
                            termination = "capture"
                if termination == "update_limit" and (terminated or truncated):
                    termination = "environment_terminated"
                if termination != "update_limit":
                    break
            applied += 1
            record["execution_complete"] = True
            write_json(directory / "attempts" / f"k{update:03d}.json", record)
            if getattr(args, "checkpoint_updates", False):
                save_trajectory(directory, history, targets)
            pilot_event(stage="between_updates", completed_updates=applied, timeout_seconds=120.)
            if termination != "update_limit":
                break
        arrays = {key: np.asarray(values) for key, values in history.items()}
        arrays["targets"] = targets
        np.savez_compressed(directory / "trajectory.npz", **arrays)
        target_indices = np.arange(pursuers, n) if pursuers else np.arange(n)
        errors = np.linalg.norm(arrays["states"][:, target_indices, :3]-targets[None], axis=2)
        speeds = np.linalg.norm(arrays["states"][:, target_indices, 3:], axis=2)
        reached = (errors <= .25) & (speeds <= .3)
        pair_distance = np.stack([np.linalg.norm(arrays["states"][:, i, :3]-arrays["states"][:, j, :3], axis=1)
                                   for i, j in combinations(range(n), 2)], axis=1)
        summary = dict(scenario=name, variant=variant, termination=termination, applied_updates=applied,
                       equilibrium_method=args.equilibrium_method,
                       equilibrium_accepted_updates=sum(bool(r.get("validation", {}).get("accepted", False)) for r in attempts if r["applied"]),
                       candidate_repair_restarts=sum(len(r.get("repair_history", [])) for r in attempts),
                       maximum_recorded_kkt_norm=max((r["validation"]["kkt_norm"] for r in attempts if "validation" in r), default=None),
                       simulated_duration_s=float(arrays["times"][-1]), final_goal_distance_m=errors[-1],
                       initial_goal_distance_m=errors[0], goal_progress_m=errors[0]-errors[-1],
                       final_goal_speed_mps=speeds[-1], ever_reached_with_low_speed=np.any(reached, axis=0),
                       first_goal_stop_time_s=[float(arrays["times"][np.flatnonzero(reached[:, i])[0]])
                                              if reached[:, i].any() else None for i in range(len(target_indices))],
                       minimum_measured_pair_distance_m=float(pair_distance.min()),
                       minimum_measured_altitude_m=float(arrays["states"][:, :, 2].min()),
                       maximum_measured_altitude_m=float(arrays["states"][:, :, 2].max()),
                       maximum_measured_speed_mps=float(np.linalg.norm(arrays["states"][:, :, 3:], axis=2).max()),
                       motor_saturation_fraction=float(arrays["motor_saturation"][1:].mean()) if applied else None,
                       force_limited_fraction=float(arrays["force_limited"][1:].mean()) if applied else None,
                       reference_position_rmse_m=float(np.sqrt(np.mean(np.sum((arrays["states"][1:, :, :3]-arrays["references"][1:, :, :3])**2, axis=2)))) if applied else None,
                       reference_velocity_rmse_mps=float(np.sqrt(np.mean(np.sum((arrays["states"][1:, :, 3:]-arrays["references"][1:, :, 3:])**2, axis=2)))) if applied else None,
                       median_solve_seconds=float(np.median([r["compute_time"] for r in attempts])),
                       preparation_seconds=prepared, wall_seconds=perf_counter()-started,
                       capture_events=capture_events,
                       costs_at_last_input=attempts[-1]["costs"], cost_terms_at_last_input=attempts[-1]["cost_terms"],
                       interpretation="Matched local primal-controller diagnostic; not a global-equilibrium or real-time guarantee")
        write_json(directory / "summary.json", summary)
        return summary
    finally:
        env.close()


def save_trajectory(directory, history, targets):
    """Atomic prefix checkpoint; a killed worker cannot erase completed ticks."""
    temporary = directory / "trajectory.tmp.npz"
    np.savez_compressed(temporary, targets=targets, **{key: np.asarray(value) for key, value in history.items()})
    os.replace(temporary, directory / "trajectory.npz")
    states = np.asarray(history["states"])
    errors = np.linalg.norm(states[:, :, :3]-targets[None], axis=2)
    distances = [np.linalg.norm(states[:, i, :3]-states[:, j, :3], axis=1)
                 for i, j in combinations(range(states.shape[1]), 2)]
    write_json(directory / "trajectory_progress.json", dict(simulated_duration_s=history["times"][-1],
        initial_goal_distance_m=errors[0], final_goal_distance_m=errors[-1],
        goal_progress_m=errors[0]-errors[-1], minimum_measured_pair_distance_m=float(np.min(distances)),
        goal_distance_trace_m=errors[::12], goal_trace_time_s=np.asarray(history["times"])[::12]))


def render(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    scenarios = list(dict.fromkeys(s["scenario"] for s in summaries))
    variants = list(dict.fromkeys(s["variant"] for s in summaries))
    colors = ["#1768ac", "#d97706", "#24845b", "#b45b31", "#8844a6"]
    fig, axes = plt.subplots(len(scenarios), len(variants), figsize=(11, 3.3*len(scenarios)), squeeze=False,
                             layout="constrained")
    inputs = []
    for summary in summaries:
        row, col = scenarios.index(summary["scenario"]), variants.index(summary["variant"])
        directory = output/f"{summary['scenario']}_{summary['variant']}"
        config = json.loads((directory/"config.json").read_text())
        with np.load(directory/"trajectory.npz") as data:
            agents = range(config["num_pursuers"], config["num_agents"]) if config["num_pursuers"] else range(config["num_agents"])
            for local, agent in enumerate(agents):
                error = np.linalg.norm(data["states"][:, agent, :3]-data["targets"][local], axis=1)
                label = f"Evader {local+1}" if config["num_pursuers"] else f"Agent {local+1}"
                axes[row, col].plot(data["times"], error, label=label, color=colors[local])
        axis = axes[row, col]
        axis.axhline(.25, color="#777777", linestyle=":", linewidth=.8)
        axis.set(title=f"{summary['scenario']} · {summary['variant']}\n{summary['termination']}",
                 xlabel="Simulated time [s]", ylabel="Assigned-goal distance [m]")
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
        inputs.extend([directory/"trajectory.npz", directory/"summary.json", directory/"config.json"])
    for extension in ("png", "pdf"):
        fig.savefig(output/f"goal_comparison.{extension}", dpi=170)
    plt.close(fig)
    for name in scenarios:
        tracks = []
        for variant in variants:
            directory = output/f"{name}_{variant}"
            config = json.loads((directory/"config.json").read_text())
            with np.load(directory/"trajectory.npz") as data:
                summary = json.loads((directory/"summary.json").read_text())
                tracks.append(dict(variant=variant, config=config, summary=summary,
                                   times=data["times"], states=data["states"], targets=data["targets"]))
        points = np.vstack([t["states"][:, :, :3].reshape(-1, 3) for t in tracks]+[t["targets"] for t in tracks])
        low, high = points.min(axis=0), points.max(axis=0)
        padding = np.maximum(.1*(high-low), .3)
        fig = plt.figure(figsize=(12, 5.8))
        fig.subplots_adjust(left=.015, right=.94, bottom=.14, top=.82, wspace=.23)
        artists = []
        for col, track in enumerate(tracks):
            axis = fig.add_subplot(1, len(tracks), col+1, projection="3d")
            axis.set(xlim=(low[0]-padding[0], high[0]+padding[0]), ylim=(low[1]-padding[1], high[1]+padding[1]),
                     zlim=(max(0., low[2]-padding[2]), high[2]+padding[2]), title=track["variant"],
                     xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
            axis.set_box_aspect((1.3, 1, .8))
            axis.view_init(elev=28, azim=-65)
            lines, heads = [], []
            for agent in range(track["config"]["num_agents"]):
                color = colors[agent % len(colors)]
                pursuers = track["config"]["num_pursuers"]
                agent_label = (f"P{agent+1}" if agent < pursuers else f"E{agent-pursuers+1}") if pursuers else f"A{agent+1}"
                lines.append(axis.plot([], [], [], color=color, linewidth=1.5, label=agent_label)[0])
                heads.append(axis.plot([], [], [], "o", color=color, markersize=5)[0])
            for local, target in enumerate(track["targets"]):
                agent = local+track["config"]["num_pursuers"]
                axis.scatter(*target, marker="*", s=45, color=colors[agent % len(colors)])
            label = axis.text2D(.01, .94, "", transform=axis.transAxes, fontsize=8)
            artists.append((lines, heads, label))
        duration = max(t["times"][-1] for t in tracks)
        fig.legend(artists[0][0], [line.get_label() for line in artists[0][0]], loc="lower center",
                   bbox_to_anchor=(.5, .075), ncol=len(artists[0][0]), fontsize=8, frameon=False)
        caption = fig.suptitle(name, fontsize=12)
        fig.supxlabel("PyBullet measured trajectories · stars mark assigned goals · planning pauses simulated time\n"
                      "STOP marks the last recorded state; no flight is simulated afterwards", fontsize=8)
        def update(clock):
            caption.set_text(f"{name}: prediction horizon, move blocking and terminal cost · t={clock:.2f} s")
            for track, (lines, heads, label) in zip(tracks, artists):
                index = min(np.searchsorted(track["times"], clock, side="right")-1, len(track["times"])-1)
                index = max(index, 0)
                for agent, (line, head) in enumerate(zip(lines, heads)):
                    trajectory = track["states"][:index+1, agent, :3]
                    line.set_data_3d(*trajectory.T)
                    head.set_data_3d(*[[v] for v in trajectory[-1]])
                stopped = clock >= track["times"][-1]-1e-9
                status = f"\nSTOP: {track['summary']['termination']}" if stopped else ""
                goals = []
                for local, target in enumerate(track["targets"]):
                    agent = local+track["config"]["num_pursuers"]
                    distance = np.linalg.norm(track["states"][index, agent, :3]-target)
                    speed = np.linalg.norm(track["states"][index, agent, 3:])
                    prefix = "E" if track["config"]["num_pursuers"] else "A"
                    goals.append(f"{prefix}{local+1}: d={distance:.2f} m, v={speed:.2f} m/s")
                label.set_text(f"t={track['times'][index]:.2f} s; "
                               f"H={track['config']['horizon']}, blocks={track['config']['control_blocks']}\n"+
                               "\n".join(goals)+status)
                label.set_color("#a32020" if stopped else "black")
        clocks = np.minimum(np.arange(int((duration+1)*12))/12, duration)
        animation = FuncAnimation(fig, update, frames=clocks, blit=False)
        animation.save(output/f"{name}.mp4", writer=FFMpegWriter(fps=12, codec="libx264",
            extra_args=["-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]), dpi=100)
        plt.close(fig)
    renderer = output/"plot_sources"/Path(__file__).name
    renderer.parent.mkdir(exist_ok=True)
    renderer.write_bytes(Path(__file__).read_bytes())
    write_json(output/"plot_manifest.json", dict(
        renderer_source=str(renderer.relative_to(output)), renderer_sha256=hashlib.sha256(renderer.read_bytes()).hexdigest(),
        inputs={str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                   if p.suffix in (".png", ".pdf", ".mp4")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("custom", "adversarial-goal-tracking"), default="custom",
                        help="Named tested configuration; explicitly supplied options override its defaults")
    parser.add_argument("--scenarios", nargs="+", choices=("cooperative", "adversarial"), default=["cooperative", "adversarial"])
    parser.add_argument("--cooperative-layout", choices=("head-on", "all-moving"), default="head-on",
                        help="all-moving sends agent 3 from y=2 to y=-2 while retaining the original encounter")
    parser.add_argument("--variants", nargs="+", choices=("baseline", "blocked", "terminal"), default=["baseline", "blocked", "terminal"])
    parser.add_argument("--blocks", type=int, nargs="+", default=[1, 1, 2, 4, 6, 6])
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--velocity-weight", type=float, default=0.)
    parser.add_argument("--effort-weight", type=float, default=.1, help="Cooperative acceleration cost coefficient")
    parser.add_argument("--terminal-scale", type=float, default=1.)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--response-sweeps", type=int, default=3)
    parser.add_argument("--response-fractions", type=float, nargs="+", default=[.7, .35, 1.],
                        help="Fractions of each proposed player update, tested in this order")
    parser.add_argument("--goal-weight", type=float, default=2.)
    parser.add_argument("--evasion-weight", type=float, default=50.)
    parser.add_argument("--input-bound", type=float, default=5.)
    parser.add_argument("--speed-bound", type=float)
    parser.add_argument("--jerk-bound", type=float)
    parser.add_argument("--physical-speed-bound", type=float)
    parser.add_argument("--emergency-jerk-bound", type=float, default=20.)
    parser.add_argument("--pid-mode", choices=("legacy", "feedforward", "limited"), default="legacy")
    parser.add_argument("--equilibrium-method", choices=("potential", "gne", "vgne"), default="potential")
    parser.add_argument("--solver-backend", choices=("primal",), default="primal")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--reserve-retry", action="store_true", help="Retry infeasible tightened plans with physical altitude bounds; keep all execution checks")
    parser.add_argument("--render-only", action="store_true", help="Render saved cases without rerunning any numerical experiments")
    args = parser.parse_args()
    from tools.reproduce import writable_output
    writable_output(args.output)
    apply_preset(args, sys.argv[1:])
    if args.solver_backend == "nashopt" and (args.scenarios != ["cooperative"] or args.equilibrium_method == "potential"):
        parser.error("NashOpt requires --scenarios cooperative and --equilibrium-method gne|vgne")
    if args.render_only:
        render(args.output, json.loads((args.output/"summary.json").read_text())["cases"])
        return 0
    if args.updates <= 0 or args.max_iterations <= 0 or args.response_sweeps <= 0:
        parser.error("Iteration counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    source_names = ["src/game_theory_mpc/blocked_game_mpc.py", "src/game_theory_mpc/cooperative_equilibrium_mpc.py",
                    "src/game_theory_mpc/reusable_gne.py",
                    "src/game_theory_mpc/acceleration_pid.py", "experiments/experiment_blocked_terminal_mpc.py",
                    "experiments/equilibrium_mpc_execution.py", "experiments/equilibrium_selection/archive.py"]
    hashes = {}
    for name in source_names:
        content = (ROOT/name).read_bytes()
        destination = args.output/"source_snapshots"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    write_json(args.output/"manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes, command=[sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        library_versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "jax", "jaxlib", "nashopt", "pybullet", "gym-pybullet-drones")},
        comparison="Same primal solver within each scenario; stage costs unchanged by default; blocked vs terminal differs only by Riccati terminal cost",
        stage_velocity_weight=args.velocity_weight,
        scope="Separate matched model/criterion experiment, not a new result for the frozen KKT timing benchmark"))
    results = []
    for name in args.scenarios:
        for variant in args.variants:
            results.append(run_case(name, variant, args, args.output))
            write_json(args.output/"summary.json", dict(complete=False, cases=results))
    write_json(args.output/"summary.json", dict(complete=True, cases=results))
    lines = ["# Move blocking and terminal-cost comparison", "", "Same primal solver within each scenario; airborne start. No global equilibrium or real-time claim.", "",
             "| Scenario | Variant | Duration [s] | Stop | Final goal distances [m] | Final goal speeds [m/s] | Median solve [s] |",
             "|---|---|---:|---|---|---|---:|"]
    for case in results:
        lines.append(f"| {case['scenario']} | {case['variant']} | {case['simulated_duration_s']:.3f} | {case['termination']} | "
                     f"{', '.join(f'{x:.3f}' for x in case['final_goal_distance_m'])} | "
                     f"{', '.join(f'{x:.3f}' for x in case['final_goal_speed_mps'])} | {case['median_solve_seconds']:.4g} |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    if not args.no_render:
        render(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
