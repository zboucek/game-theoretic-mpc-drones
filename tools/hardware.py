# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Offline reanalysis of the accepted five-drone log; requires no ROS/hardware."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def analyze(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    frames = defaultdict(list)
    for row in rows:
        frames[row['t']].append(row)
    names = sorted({r['drone'] for r in rows})
    if len(names) != 5 or any(sorted(r['drone'] for r in f) != names for f in frames.values()):
        raise ValueError('Every timestamp must contain exactly the five logged vehicles')
    controls = [frame for frame in frames.values() if frame[0]['phase'].startswith('mpc')]
    if not controls:
        raise ValueError('Missing MPC phase')
    positions = lambda frame: np.asarray([[float(r['raw_' + a]) for a in 'xyz'] for r in frame])
    distances = []
    for frame in frames.values():
        p = positions(frame)
        distances.extend(np.linalg.norm(p[i] - p[j]) for i in range(5) for j in range(i + 1, 5))
    control_rows = [frame[0] for frame in controls]
    solve = np.array([float(r['solve_time_s']) for r in control_rows])
    loop = np.array([float(r['loop_time_s']) for r in control_rows])
    centroids = np.array([positions(f).mean(axis=0) for f in controls])
    errors = np.array([[float(r['raw_' + a]) - float(r['cmd_' + a]) for a in 'xyz']
                       for frame in controls for r in frame])
    formation_errors = []
    for frame in controls:
        p = positions(frame)
        target = np.array([[float(r['target_' + a]) for a in 'xyz'] for r in frame])
        diff = [np.linalg.norm(p[i] - p[j]) - np.linalg.norm(target[i] - target[j])
                for i in range(5) for j in range(i + 1, 5)]
        formation_errors.append(np.sqrt(np.mean(np.square(diff))))
    last = list(frames.values())[-1]
    landing_errors = [np.linalg.norm([float(r['raw_' + a]) - float(r['target_' + a]) for a in 'xy'])
                      for r in last]
    metrics = {
        'mpc_steps': len(controls),
        'max_solver_ms': float(solve.max() * 1000),
        'p95_solver_ms': float(np.percentile(solve, 95) * 1000),
        'max_loop_ms': float(loop.max() * 1000),
        'p95_loop_ms': float(np.percentile(loop, 95) * 1000),
        'min_pair_distance_m': float(min(distances)),
        'centroid_span_m': np.ptp(centroids, axis=0).tolist(),
        'command_tracking_rmse_m': float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
        'p95_pair_rmse_to_commanded_geometry_m': float(np.percentile(formation_errors, 95)),
        'max_final_landing_xy_error_m': float(max(landing_errors)),
    }
    return metrics, rows


def plot(rows, output_base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_drone = defaultdict(list)
    for row in rows:
        by_drone[row['drone']].append(row)
    fig, (xy, timing) = plt.subplots(1, 2, figsize=(7.2, 3.15))
    for name, samples in by_drone.items():
        p = np.array([[float(r['raw_' + a]) for a in 'xyz'] for r in samples])
        line, = xy.plot(p[:, 0], p[:, 1], label=name, linewidth=1.5)
        xy.scatter(*p[0, :2], color=line.get_color(), s=20)
        xy.scatter(*p[-1, :2], color=line.get_color(), marker='x', s=25)
    xy.set(xlabel='X [m]', ylabel='Y [m]', title='Measured hardware trajectories')
    xy.axis('equal')
    xy.legend(fontsize=7, ncol=2)
    unique = {r['t']: r for r in rows if r['phase'].startswith('mpc')}
    for field, label in [('solve_time_s', 'NashLinearMPC solve'), ('loop_time_s', 'Control loop')]:
        timing.plot([float(r[field]) * 1000 for r in unique.values()], label=label, linewidth=1)
    timing.axhline(80, color='tab:orange', linestyle='--', label='80 ms solve gate')
    timing.axhline(100, color='tab:red', linestyle=':', label='100 ms loop budget')
    timing.set(xlabel='MPC step', ylabel='Wall-clock time [ms]', title='Real-time execution')
    timing.legend(fontsize=7)
    for ax in (xy, timing):
        ax.grid(alpha=.25)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(Path(output_base).with_suffix('.' + ext), bbox_inches='tight', dpi=180)
    plt.close(fig)
