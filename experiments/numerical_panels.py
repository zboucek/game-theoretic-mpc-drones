# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Illustrate archived solution diagnostics and weight tuning without new solves."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-theory-mpc-manuscript-plots")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/game-theory-mpc-manuscript-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("gne", "#1768ac", "-", "Joint KKT: GNE"),
           ("vgne", "#bd5b00", "--", "Joint KKT: vGNE"),
           ("potential", "#26814a", "-.", "Potential minimization"))
AGENT_COLORS = ("#1768ac", "#d97706", "#24845b")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class Inputs:
    def __init__(self):
        self.hashes = {}

    def record(self, path):
        self.hashes[os.path.relpath(path, ROOT)] = digest(path)

    def json(self, path):
        self.record(path)
        return json.loads(path.read_text())

    def trajectory(self, path):
        self.record(path)
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}


def diagnostic_data(source, inputs):
    plan = inputs.json(source / "plan_confirm.json")
    case = next(c for c in plan["instances"]
                if c["geometry"] == "head_on" and c["perturbation"] == 0)
    initial, targets = np.asarray(case["states"]), np.asarray(case["targets"])
    moving = np.linalg.norm(initial[:, :3] - targets, axis=1) > .25
    directions = targets[moving, :2] - initial[moving, :2]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    np.testing.assert_array_equal(directions, [[1., 0.], [-1., 0.]])
    series, findings = {}, {}
    for method, *_ in METHODS:
        episode = inputs.json(source / "episodes/confirm" / f"head_on-p0-braking-{method}.json")
        states, times = np.asarray(episode["applied_states"]), np.asarray(episode["simulated_time_s"])
        np.testing.assert_array_equal(states[0], initial)
        np.testing.assert_allclose(times, np.arange(61) * plan["protocol"]["dt"], atol=1e-12)
        np.testing.assert_allclose(states[:, :, 2], 1., atol=1e-8)
        assert episode["accepted_updates"] == 60 and episode["termination"] == "update_limit"
        for update, attempt_id in enumerate(episode["attempt_ids"]):
            record = inputs.json(source / "raw" / f"{attempt_id}.json")
            np.testing.assert_array_equal(record["targets"], targets)
            np.testing.assert_array_equal(record["states"], states[update])
        route_coordinates = np.einsum("tij,ij->ti", states[:, moving, :2], directions)
        worst = route_coordinates.min(axis=1)
        passage = np.flatnonzero(worst >= .25)
        first = float(times[passage[0]]) if len(passage) else None
        assert (first is not None) == episode["conflict_ever_passed"]
        assert first == episode["first_conflict_passage_s"]
        series[method] = (states, times, worst)
        findings[method] = dict(accepted_updates=60, first_conflict_passage_s=first,
                                minimum_route_coordinate_m=worst.tolist())
        if method == "potential":
            # Check overshoot and subsequent reversal directly in the executed
            # states, rather than inferring target settling from passage.
            overshoots, reversal_times = [], []
            for agent, direction in enumerate(directions[:, 0]):
                error = direction * (states[:, agent, 0] - targets[agent, 0])
                peak = int(np.argmax(error))
                assert error[peak] > 0 and 0 < peak < len(times) - 1
                velocity = direction * states[:, agent, 3]
                assert velocity[peak - 1] > 0 and np.any(velocity[peak:] < 0)
                overshoots.append(float(error[peak]))
                reversal_times.append(float(times[peak]))
            findings[method].update(target_overshoot_x_m=overshoots,
                                    sampled_turning_times_s=reversal_times)
    assert findings["gne"]["first_conflict_passage_s"] is None
    assert findings["vgne"]["first_conflict_passage_s"] is None
    np.testing.assert_allclose(findings["potential"]["first_conflict_passage_s"], .7)
    return series, findings, targets


def rollout_metrics(trajectory, arrival):
    times, states, controls, targets = (trajectory[k] for k in ("times", "states", "controls", "targets"))
    error2 = np.sum((states[:, :, :3] - targets[None]) ** 2, axis=2)
    dt = np.diff(times)
    integrated = float(np.sum(.5 * (error2[:-1].sum(axis=1) + error2[1:].sum(axis=1)) * dt))
    effort = float(np.sum(np.sum(controls ** 2, axis=(1, 2)) * dt))
    arrived = np.all((error2 <= arrival["distance"] ** 2)
                     & (np.linalg.norm(states[:, :, 3:], axis=2) <= arrival["speed"]), axis=1)
    joint = None
    for i in np.flatnonzero(arrived):
        end = np.searchsorted(times, times[i] + arrival["dwell_s"] - 1e-9)
        if end < len(times) and np.all(arrived[i:end + 1]):
            joint = float(times[i])
            break
    assert joint is not None
    return dict(integrated_position_error_m2s=integrated, input_effort_m2_s3=effort, joint_arrival_s=joint)


def tuning_data(source, inputs):
    protocol = inputs.json(source / "manifest.json")["protocol"]
    frozen = inputs.json(source / "frozen_weights.json")
    assert frozen["frozen_before_test"]
    np.testing.assert_array_equal(frozen["baseline"], [1, 1, 1])
    np.testing.assert_array_equal(frozen["full"], [1, .3, 1])
    assert protocol["effort_weight"] == .1
    pairs = []
    for case in protocol["cases"]["test"]:
        values = []
        for profile in ("baseline", "full"):
            directory = source / "test" / profile / case["id"]
            summary = inputs.json(directory / "summary.json")
            trajectory = inputs.trajectory(directory / "trajectory.npz")
            assert summary["success"] and summary["completed"]
            np.testing.assert_allclose(trajectory["states"][0], case["states"])
            np.testing.assert_array_equal(trajectory["targets"], case["targets"])
            np.testing.assert_allclose(trajectory["times"][-1], 10.)
            measured = rollout_metrics(trajectory, protocol["arrival"])
            for field, value in measured.items():
                np.testing.assert_allclose(value, summary[field], rtol=1e-12, atol=1e-12)
            values.append(measured)
        manual, tuned = values
        pair = dict(case=case["id"], family=case["family"],
                    error_change_percent=100 * (tuned["integrated_position_error_m2s"] / manual["integrated_position_error_m2s"] - 1),
                    effort_change_percent=100 * (tuned["input_effort_m2_s3"] / manual["input_effort_m2_s3"] - 1),
                    arrival_change_s=tuned["joint_arrival_s"] - manual["joint_arrival_s"])
        pairs.append(pair)
    assert len(pairs) == 12 and sum(p["error_change_percent"] > 0 for p in pairs) == 2
    assert all(p["effort_change_percent"] > 0 and p["arrival_change_s"] < 0 for p in pairs)
    medians = {field: float(np.median([p[field] for p in pairs]))
               for field in ("error_change_percent", "effort_change_percent", "arrival_change_s")}
    return pairs, medians


def save_figure(fig, output, name):
    paths = []
    for extension in ("pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=220, metadata={"Creator": Path(__file__).name})
        paths.append(path)
    plt.close(fig)
    return paths


def plot_diagnostic(series, targets, output):
    fig = plt.figure(figsize=(7.16, 2.7), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1., 1., 1.22),
                          height_ratios=(2.1, 1.))
    axes = [fig.add_subplot(grid[0, column]) for column in range(2)]
    overviews = [fig.add_subplot(grid[1, column]) for column in range(2)]
    axes.append(fig.add_subplot(grid[:, 2]))
    detail_xlim, detail_ylim = (-.75, 4.25), (-1.1, 1.1)
    display_end_s = 6.
    agent_styles = ("-", "--", ":")
    for axis, overview, method in zip(axes[:2], overviews, ("gne", "potential")):
        states, times, _ = series[method]
        shown = times <= display_end_s + 1e-12
        displayed = states[shown]
        # Both views use the full episode; the upper view only crops space.
        # The overview retains the entire trajectories, including reversals.
        for agent, (color, style) in enumerate(zip(AGENT_COLORS[:2], agent_styles[:2])):
            path = displayed[:, agent, :2]
            axis.plot(*path.T, color=color, ls=style, lw=.8, zorder=2)
            axis.scatter(*path[0], marker="o", s=19, facecolor="white", edgecolor=color,
                         linewidth=.9, zorder=6)
            axis.scatter(*path[-1], marker="s", s=14, color=color, zorder=5)
            offset = ((-16, 7), (3, 7), (5, 1))[agent]
            axis.annotate(f"A{agent + 1}", path[0], xytext=offset,
                          textcoords="offset points", fontsize=7, color=color)
            if agent < 2:
                # Keep the GNE arrows outside the region in which A1/A2 overlap.
                first, last = ((1, 5) if agent == 0 else (7, 9)) if method == "gne" else (4, 7)
                axis.annotate("", xy=path[last], xytext=path[first],
                              arrowprops=dict(arrowstyle="-|>", color=color, lw=.7,
                                              mutation_scale=8, shrinkA=0, shrinkB=0), zorder=7)
        axis.axvline(0, color=".8", lw=.6, ls=":", zorder=0)
        axis.set(xlim=detail_xlim, ylim=detail_ylim, ylabel="$y$ [m]",
                 xticks=[0, 2, 4], yticks=[-1, 0, 1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_anchor("S")
        # The rectangle locates the upper view in the full-episode paths.
        overview.add_patch(Rectangle((detail_xlim[0], detail_ylim[0]),
                                     np.diff(detail_xlim)[0], np.diff(detail_ylim)[0],
                                     facecolor=".95", edgecolor=".5", lw=.6, zorder=0))
        for agent, (color, style) in enumerate(zip(AGENT_COLORS, agent_styles)):
            path = displayed[:, agent, :2]
            overview.plot(*path.T, color=color, ls=style, lw=.7, zorder=1)
            overview.scatter(*displayed[0, agent, :2], marker="o", s=10,
                             facecolor="white", edgecolor=color, linewidth=.7, zorder=2)
            overview.scatter(*path[-1], marker="s", s=11, color=color, zorder=4)
            overview.scatter(*targets[agent, :2], marker="*", s=36,
                             color=color, linewidth=.5, zorder=3)
            offset = ((-3, 5), (3, 5), (7, 0))[agent]
            overview.annotate(f"A{agent + 1}", targets[agent, :2], xytext=offset,
                              textcoords="offset points", ha="right" if agent == 0 else "left",
                              fontsize=6, color=color)
            if method == "potential" and agent < 2:
                # Show outgoing and returning motion at separate positions.
                for first, last in ((17, 21), (45, 49)):
                    overview.annotate("", xy=path[last], xytext=path[first],
                                      arrowprops=dict(arrowstyle="-|>", color=color, lw=.6,
                                                      mutation_scale=6, shrinkA=0, shrinkB=0), zorder=5)
        overview.set(xlim=(-11, 17), ylim=(-1.5, 3.5), xlabel="$x$ [m]", ylabel="$y$ [m]",
                     xticks=[-10, 0, 10], yticks=[0, 2])
        overview.set_aspect("equal", adjustable="box")
        overview.set_anchor("N")
        overview.tick_params(labelsize=6.5, pad=2)
        overview.xaxis.label.set_size(7.5)
        overview.yaxis.label.set_size(7.5)
        axis.tick_params(labelsize=6.5, pad=2)
        axis.yaxis.label.set_size(7.5)

    method_colors = ("#303030", "#888888", "#7953a3")
    method_styles = ("-", "--", "-.")
    for (method, *_), color, style in zip(METHODS, method_colors, method_styles):
        _, times, worst = series[method]
        axes[2].plot(times, worst, color=color, ls=style, lw=1.)
    axes[2].axhline(.25, color="#9b3030", ls=":", lw=.9)
    axes[2].annotate("Both passed", xy=(.1, .25), xytext=(0, 4),
                     textcoords="offset points", ha="left", fontsize=6.5, color="#9b3030")
    _, times, worst = series["potential"]
    first = np.flatnonzero(worst >= .25)[0]
    axes[2].scatter(times[first], worst[first], s=15, color=method_colors[2], zorder=5)
    axes[2].annotate("Potential\nFirst passage: 0.7 s", xy=(times[first], worst[first]),
                     xytext=(.2, 6.), color=method_colors[2],
                     arrowprops=dict(arrowstyle="->", color=method_colors[2], lw=.7), fontsize=6.5)
    axes[2].set(xlim=(0, 6), ylim=(-5, 10.5), xticks=[0, 2, 4, 6], yticks=[-4, 0, 4, 8],
                xlabel="Simulated time [s]", ylabel="Least progress past $x=0$ [m]")
    axes[2].grid(alpha=.2)
    axes[2].set_box_aspect(.72)
    # Identify nearly coincident curves at their actual data coordinates,
    # rather than asking readers to decode another line-style legend.
    for method, label, color, anchor_t, text_position in (
            ("gne", "GNE", method_colors[0], 4.5, (3., -4.4)),
            ("vgne", "vGNE", method_colors[1], 5., (3., -1.1))):
        _, times, progress = series[method]
        axes[2].annotate(label, xy=(anchor_t, np.interp(anchor_t, times, progress)),
                         xytext=text_position, color=color, fontsize=6.5, va="center",
                         arrowprops=dict(arrowstyle="-", color=color, lw=.65))
    for axis in axes[2:]:
        axis.tick_params(labelsize=6.5, pad=2)
        axis.xaxis.label.set_size(7.)
        axis.yaxis.label.set_size(7.)
    handles = [Line2D([], [], color="black", ls="", marker="o", markerfacecolor="white", label="Start"),
                Line2D([], [], color="black", ls="", marker="s", label="6 s"),
                Line2D([], [], color="black", ls="", marker="*", label="Target")]
    legend = fig.legend(handles=handles, loc="outside upper center", bbox_to_anchor=(.29, 1.),
                        ncols=3, frameon=False,
                        fontsize=6.5, handlelength=1.6, columnspacing=1., borderaxespad=0.)

    def compact_spatial_views():
        # Equal-unit axes leave slack inside their grid cells. Pack the
        # overview against the detail's tick labels with a small clear gap.
        renderer = fig.canvas.get_renderer()
        for detail, overview in zip(axes[:2], overviews):
            target_top = detail.get_tightbbox(renderer).y0 - .06 * fig.dpi
            shift = (target_top - overview.get_tightbbox(renderer).y1) / fig.bbox.height
            overview.set_position(overview.get_position().translated(0, shift))
        fig.canvas.draw()

    # Export compact panels at their own tight bounds. The spatial panels
    # share equal-unit limits; their height is not inherited from panel (c).
    fig.canvas.draw()
    fig.set_layout_engine("none")
    compact_spatial_views()
    renderer = fig.canvas.get_renderer()
    key_height = legend.get_window_extent(renderer).height / fig.bbox.height
    spatial_top = max(axis.get_position().y1 for axis in axes[:2])
    spatial_center = .5 * (axes[0].get_position().x0 + axes[1].get_position().x1)
    legend.set_bbox_to_anchor((spatial_center, spatial_top + .012 + key_height))
    fig.canvas.draw()
    # Standalone previews also put labels below; the article uses only the
    # raw panel PDFs above and supplies its own subfloat captions.
    captions = ("(a) GNE paths\n0–6 s", "(b) Potential minimization\n0–6 s",
                "(c) Conflict passage\n0–6 s")
    for axis in fig.axes:
        axis.set_visible(True)
    for axis, caption in zip((*overviews, *axes[2:]), captions):
        axis.set_xlabel(axis.get_xlabel() + "\n" + caption, fontsize=7.)
    for overview in overviews:
        overview.set_in_layout(True)
    fig.set_layout_engine("constrained")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    compact_spatial_views()
    key_height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
    spatial_top = max(axis.get_position().y1 for axis in axes[:2])
    spatial_center = .5 * (axes[0].get_position().x0 + axes[1].get_position().x1)
    legend.set_bbox_to_anchor((spatial_center, spatial_top + .012 + key_height))
    return save_figure(fig, output, "numerical_conflict_example")


def plot_article_diagnostic(series, targets, output):
    """Render native-size article panels, independently of the four-panel preview."""
    # Match main.tex's 7-inch text width and 67%/31% spatial/passage split.
    # Fixed canvases preserve one physical font/marker size after inclusion.
    spatial_width, passage_width, height = 7 * .67 * .49, 7 * .31, 2.12
    detail_xlim, detail_ylim = (-2.5, 2.5), (-1.2, 1.2)
    overview_xlim, overview_ylim = (-11, 17), (-1.5, 3.5)
    styles = ("-", "--", ":")
    outputs = []

    def axis_at(fig, left, bottom, width, axis_height):
        canvas_width, canvas_height = fig.get_size_inches()
        axis = fig.add_axes([left / canvas_width, bottom / canvas_height,
                             width / canvas_width, axis_height / canvas_height])
        axis.tick_params(labelsize=7.5, pad=2, length=3, width=.7)
        for spine in axis.spines.values():
            spine.set_linewidth(.7)
        return axis

    def direction(axis, path, first, last, color):
        axis.annotate("", xy=path[last], xytext=path[first],
                      arrowprops=dict(arrowstyle="-|>", color=color, lw=.85,
                                      mutation_scale=8, shrinkA=0, shrinkB=0), zorder=7)

    for name, method in (("a", "gne"), ("b", "potential")):
        fig = plt.figure(figsize=(spatial_width, height))
        width = spatial_width - .51
        detail = axis_at(fig, .43, 1.10, width, width * 2.4 / 5)
        overview = axis_at(fig, .43, .40, width, width * 5 / 28)
        states, times, _ = series[method]
        displayed = states[times <= 6. + 1e-12]
        for agent, (color, style) in enumerate(zip(AGENT_COLORS, styles)):
            path = displayed[:, agent, :2]
            if agent < 2:
                detail.plot(*path.T, color=color, ls=style, lw=.9)
                detail.scatter(*path[0], marker="o", s=19, facecolor="white",
                               edgecolor=color, linewidth=.9, zorder=6)
                detail.scatter(*path[-1], marker="s", s=14, color=color, zorder=5)
                detail.annotate(f"A{agent + 1}", path[0],
                                xytext=((-19, 7) if agent == 0 else (8, 7)),
                                textcoords="offset points", fontsize=8, color=color)
                first, last = ((8, 11) if agent == 0 else (14, 17)) if method == "gne" else (6, 9)
                direction(detail, path, first, last, color)
            overview.plot(*path.T, color=color, ls=style, lw=.85)
            overview.scatter(*path[0], marker="o", s=11, facecolor="white",
                             edgecolor=color, linewidth=.75, zorder=3)
            overview.scatter(*path[-1], marker="s", s=12, color=color, zorder=4)
            overview.scatter(*targets[agent, :2], marker="*", s=36, color=color,
                             linewidth=.5, zorder=3)
            overview.annotate(f"A{agent + 1}", targets[agent, :2],
                              xytext=((-3, 5), (3, 5), (7, 0))[agent],
                              textcoords="offset points", fontsize=7.5, color=color,
                              ha="right" if agent == 0 else "left")
            if method == "potential" and agent < 2:
                for first, last in ((17, 21), (45, 49)):
                    direction(overview, path, first, last, color)
        detail.axvline(0, color=".8", lw=.6, ls=":", zorder=0)
        detail.set(xlim=detail_xlim, ylim=detail_ylim, xticks=[-2, 0, 2],
                   yticks=[-1, 0, 1], ylabel="$y$ [m]")
        overview.add_patch(Rectangle((detail_xlim[0], detail_ylim[0]), 5, 2.4,
                                     facecolor=".95", edgecolor=".5", lw=.6, zorder=0))
        overview.set(xlim=overview_xlim, ylim=overview_ylim, xticks=[-10, 0, 10],
                     yticks=[0, 2], xlabel="$x$ [m]", ylabel="$y$ [m]")
        for axis in (detail, overview):
            axis.set_aspect("equal", adjustable="box")
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
        outputs += save_figure(fig, output, f"numerical_conflict_{name}")

    fig = plt.figure(figsize=(passage_width, height))
    axis = axis_at(fig, .48, .40, passage_width - .56, 1.59)
    colors = ("#303030", "#888888", "#7953a3")
    for (method, *_), color, style in zip(METHODS, colors, ("-", "--", "-.")):
        _, times, progress = series[method]
        axis.plot(times, progress, color=color, ls=style, lw=.9)
    axis.axhline(.25, color="#9b3030", ls=":", lw=.8)
    axis.annotate("Both passed", (.10, .25), xytext=(0, 4), textcoords="offset points",
                  fontsize=7.5, color="#9b3030")
    _, times, progress = series["potential"]
    first = np.flatnonzero(progress >= .25)[0]
    axis.scatter(times[first], progress[first], s=15, color=colors[2], zorder=5)
    axis.text(2.7, 9.7, "Potential", fontsize=7.5, color=colors[2])
    axis.annotate("0.7 s", (times[first], progress[first]),
                  xytext=(.35, 5.4), fontsize=7.5, color=colors[2],
                  arrowprops=dict(arrowstyle="->", color=colors[2], lw=.7))
    for method, label, color, anchor, position in (
            ("gne", "GNE", colors[0], 4.5, (3.8, -4.2)),
            ("vgne", "vGNE", colors[1], 5., (3.8, -1.0))):
        _, times, progress = series[method]
        axis.annotate(label, (anchor, np.interp(anchor, times, progress)),
                      xytext=position, fontsize=7.5, color=color, va="center",
                      arrowprops=dict(arrowstyle="-", color=color, lw=.65))
    axis.set(xlim=(0, 6), ylim=(-5, 10.5), xticks=[0, 2, 4, 6], yticks=[-4, 0, 4, 8],
             xlabel="Simulated time [s]", ylabel="Least progress past $x=0$ [m]")
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    axis.grid(alpha=.2)
    outputs += save_figure(fig, output, "numerical_conflict_c")

    fig = plt.figure(figsize=(7 * .67 * .4, .22))
    handles = [Line2D([], [], color="black", ls="", marker=marker, markersize=4,
                      markerfacecolor="white" if marker == "o" else "black", label=label)
               for marker, label in (("o", "Start"), ("s", "6 s"), ("*", "Target"))]
    fig.legend(handles=handles, loc="center", ncols=3, frameon=False, fontsize=7.5,
               handlelength=1., handletextpad=.4, columnspacing=.9, borderaxespad=0.)
    return outputs + save_figure(fig, output, "numerical_conflict_legend")


def plot_tuning(pairs, medians, output):
    fig, axes = plt.subplots(1, 3, figsize=(3.48, 2.15), layout="constrained")
    fields = (("error_change_percent", "Position error", "%", (-4.5, 4.5), [-4, 0, 4]),
              ("effort_change_percent", "Control effort", "%", (-2., 34.), [0, 15, 30]),
              ("arrival_change_s", "Arrival time", "s", (-.45, .05), [-.4, -.2, 0]))
    labels = [f"H{i + 1}" for i in range(6)] + [f"C{i + 1}" for i in range(6)]
    for column, (axis, (field, title, unit, limits, ticks)) in enumerate(zip(axes, fields)):
        axis.axvline(0, color=".45", ls=":", lw=.8)
        axis.axhline(5.5, color=".65", lw=.65)
        for i, pair in enumerate(pairs):
            axis.plot([0, pair[field]], [i, i], color=".55", lw=.65)
            axis.plot(pair[field], i, "o", color=".15", ms=3., zorder=3)
        axis.axvline(medians[field], color="black", ls="--", lw=.75)
        axis.set(xlim=limits, ylim=(11.65, -.65), xlabel=f"Change [{unit}]",
                 title=f"{medians[field]:+.2f}{unit}", xticks=ticks, yticks=range(12))
        axis.set_yticklabels(labels if column == 0 else [])
        axis.tick_params(labelsize=6.5, pad=2)
        axis.yaxis.set_tick_params(length=0)
        axis.xaxis.label.set_size(7.)
        axis.title.set_size(7.)
        axis.grid(axis="y", alpha=.15, lw=.5)

    # Native LaTeX supplies the common weight header and panel captions.
    # Export each panel alone, keeping a common height for aligned case rows.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [axis.get_tightbbox(renderer) for axis in axes]
    bottom, top = min(box.y0 for box in boxes), max(box.y1 for box in boxes)
    fig.set_layout_engine("none")
    for axis in axes:
        axis.set_visible(False)
    outputs = []
    for name, axis, box in zip(("a", "b", "c"), axes, boxes):
        axis.set_visible(True)
        crop = Bbox.from_extents(box.x0, bottom, box.x1, top)
        crop = crop.transformed(fig.dpi_scale_trans.inverted()).padded(.02)
        for extension in ("pdf", "png"):
            path = output / f"weight_tuning_{name}.{extension}"
            fig.savefig(path, bbox_inches=crop, dpi=220, metadata={"Creator": Path(__file__).name})
            outputs.append(path)
        axis.set_visible(False)

    # The standalone preview includes the same weights and bottom labels.
    for column, (axis, (_, title, *_)) in enumerate(zip(axes, fields)):
        axis.set_visible(True)
        axis.set_xlabel(axis.get_xlabel() + f"\n({chr(97 + column)}) {title}", fontsize=7.)
    fig.suptitle("Manual $\\longrightarrow$ tuned: $q_v:1\\longrightarrow0.3$\n"
                 "Fixed: $q_p=w_{\\mathrm{rep}}=1$, $r=0.1$", fontsize=7.)
    fig.set_layout_engine("constrained")
    fig.set_size_inches(3.48, 2.65)
    return outputs + save_figure(fig, output, "weight_tuning_design_transfer")


def generate(equilibria, weights, output):
    """Regenerate composed article figures from portable numerical data."""
    import tempfile
    import shutil
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    inputs = Inputs()
    series, findings, targets = diagnostic_data(Path(equilibria), inputs)
    pairs, medians = tuning_data(Path(weights), inputs)
    with tempfile.TemporaryDirectory() as tmp:
        temporary = Path(tmp)
        plot_diagnostic(series, targets, temporary)
        plot_tuning(pairs, medians, temporary)
        for name in ('numerical_conflict_example', 'weight_tuning_design_transfer'):
            shutil.copyfile(temporary / (name + '.pdf'), output / (name + '.pdf'))
    return dict(diagnostic=findings, tuning_pairs=pairs, tuning_medians=medians,
                input_sha256=inputs.hashes)
