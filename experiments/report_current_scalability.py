#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Render the full/blocked computation figure for article V-B.

case_rows combines raw timings with independently recomputed acceptance flags.
Rejected returned attempts stay on timing curves and use distinct markers."""
import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-theory-mpc-current-scalability")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/game-theory-mpc-manuscript-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gne", "vgne", "adversarial_br")
LABELS = dict(gne="Cooperative GNE", vgne="Cooperative vGNE", adversarial_br="Adversarial NE")
COLORS = dict(gne="#1768ac", vgne="#d97706", adversarial_br="#24845b")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def case_rows(source, manifest, summary, validation):
    audits = {r["case_id"]:r for r in validation["cases"]}
    rows = []
    for case in summary["cases"]:
        root = source/"cases"/case["id"]
        config = read(root/"config.json") if (root/"config.json").exists() else {}
        audit = {r["file"]:r for r in audits[case["id"]]["samples"]}
        records = [(path, read(path)) for path in sorted((root/"samples").glob("timed_*.json"))]
        times = [r["external_wall_seconds"] for _, r in records]
        qualities = [audit[path.name] for path, _ in records]
        eligible = [q["model_feasible"] if case["method"] == "adversarial_br" else q["numerical_accepted"] for q in qualities]
        ok_times = [t for t, ok in zip(times, eligible) if ok]
        rows.append(dict(case_id=case["id"], method=case["method"], agents=case["agents"], horizon=case["horizon"],
            parameterization=case["parameterization"], sweeps=case["sweeps"], status=case["status"],
            blocks=config.get("control_blocks"), variables=config.get("primal_variables"), inequalities=config.get("shared_inequalities"),
            timed_samples=len(times), eligible=sum(eligible), feasible=sum(q["model_feasible"] for q in qualities),
            fully_eligible=case["status"] == "complete" and len(times) == manifest["repeats"] and all(eligible),
            median_seconds=float(np.median(times)) if times else None,
            minimum_seconds=min(times) if times else None, maximum_seconds=max(times) if times else None,
            median_eligible_seconds=float(np.median(ok_times)) if ok_times else None,
            maximum_kkt_norm=max((q["kkt_norm"] for q in qualities if q["kkt_norm"] is not None), default=None),
            repairs=sum(q.get("repairs", 0) for q in qualities)))
    return rows


def curves(axis, rows, method, parameterization, sweep):
    base = 5 if method == "adversarial_br" else 3
    group = [r for r in rows if r["method"] == method and r["parameterization"] == parameterization and
             ((sweep == "agents" and r["horizon"] == 20) or (sweep == "horizon" and r["agents"] == base))]
    key = "agents" if sweep == "agents" else "horizon"
    group.sort(key=lambda r:r[key])
    color = COLORS[method]
    style, marker = ("-", "o") if parameterization == "blocked" else ("--", "s")
    xx = [r[key] for r in group]
    yy = [r["median_seconds"] if r["median_seconds"] is not None else np.nan for r in group]
    axis.plot(xx, yy, color=color, ls=style, lw=1.)
    valid = [r for r in group if r["fully_eligible"]]
    if valid:
        median = np.asarray([r["median_seconds"] for r in valid])
        axis.plot([r[key] for r in valid], median, color=color, ls="none", marker=marker, ms=3)
        axis.errorbar([r[key] for r in valid], median,
            yerr=[median-[r["minimum_seconds"] for r in valid], [r["maximum_seconds"] for r in valid]-median],
            fmt="none", ecolor=color, elinewidth=.5, capsize=1.5)
    rejected = [r for r in group if not r["fully_eligible"] and r["median_seconds"] is not None]
    if rejected:
        rejected_x = [r[key] for r in rejected]
        rejected_y = [r["median_seconds"] for r in rejected]
        axis.scatter(rejected_x, rejected_y, edgecolors=color, facecolors="white", marker=marker,
                     s=35, linewidths=.9, zorder=4)
        axis.scatter(rejected_x, rejected_y, color=color, marker="x", s=15, linewidths=.9, zorder=5)
        for row in rejected:
            right_edge = row[key] == max(xx)
            axis.annotate(dict(gne="GNE", vgne="vGNE", adversarial_br="NE")[method],
                (row[key], row["median_seconds"]), xytext=(-4 if right_edge else 4, -11 if method == "vgne" else 6),
                textcoords="offset points", ha="right" if right_edge else "left", fontsize=6.5, color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.2), zorder=6)
    for row in group:
        if row["median_seconds"] is None:
            axis.text(row[key], .95, row["status"], transform=axis.get_xaxis_transform(), rotation=90,
                      color=color, fontsize=6, va="top", ha="center")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--figures", type=Path, help="Copy manuscript assets to this directory")
    args = parser.parse_args()
    from tools.reproduce import writable_output
    if args.source is not None and not getattr(args, "check_only", False): writable_output(args.source)
    manifest, summary, validation = (read(args.source/name) for name in ("manifest.json", "summary.json", "validation.json"))
    assert summary["complete"] and validation["passed"]
    for field in ("raw_sha256", "config_sha256"):
        for name, expected in validation[field].items():
            assert digest(args.source/name) == expected, name
    rows = case_rows(args.source, manifest, summary, validation)
    pairs = []
    for method in METHODS:
        base = 5 if method == "adversarial_br" else 3
        for h in (10, 20, 30):
            group = {r["parameterization"]:r for r in rows if r["method"] == method and r["agents"] == base and r["horizon"] == h}
            if set(group) != {"full", "blocked"}:
                continue
            full, blocked = group["full"], group["blocked"]
            pairs.append(dict(method=method, agents=base, horizon=h, full=full, blocked=blocked,
                qualified_runtime_ratio=(full["median_seconds"]/blocked["median_seconds"]
                                        if full["fully_eligible"] and blocked["fully_eligible"] else None)))
    totals = dict(cases=len(rows), complete_cases=sum(r["status"] == "complete" for r in rows),
        timed_samples=sum(r["timed_samples"] for r in rows), feasible_timed=sum(r["feasible"] for r in rows),
        cooperative_timed=sum(r["timed_samples"] for r in rows if r["method"] != "adversarial_br"),
        cooperative_accepted=sum(r["eligible"] for r in rows if r["method"] != "adversarial_br"),
        adversarial_timed=sum(r["timed_samples"] for r in rows if r["method"] == "adversarial_br"),
        adversarial_feasible=sum(r["eligible"] for r in rows if r["method"] == "adversarial_br"),
        independently_checked=validation["checked_samples"], matched_pairs=validation["matched_full_blocked_pairs"])
    (args.source/"report_data.json").write_text(json.dumps(dict(totals=totals, cases=rows, pairs=pairs), indent=2)+"\n")
    fields = [k for k in rows[0] if k != "sweeps"]
    with (args.source/"cases.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size":7.5, "axes.labelsize":7.5, "axes.titlesize":7.5,
                         "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7, "pdf.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8), sharey=True, layout="constrained")
    for method in METHODS:
        curves(axes[0], rows, method, "blocked", "agents")
        for parameterization in ("blocked", "full"):
            curves(axes[1], rows, method, parameterization, "horizon")
    axes[0].set(xlabel="Drones $N$", xticks=[2,4,6,8,10], ylabel="Complete solve [s]")
    axes[1].set(xlabel="Horizon $H$", xticks=[10,20,30,40])
    for axis, caption in zip(axes, ("(a) $H=20$", "(b) $N=3/5$")):
        axis.annotate(caption, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -34),
                      textcoords="offset points", ha="center", va="top", fontsize=7.5,
                      annotation_clip=False)
    for axis in axes:
        axis.set(yscale="log")
        axis.grid(alpha=.2, which="both")
        axis.axhline(.1, color=".45", lw=.8, ls=":")
    fig.legend([Line2D([],[],color=COLORS[m],lw=1.3) for m in METHODS],
               [LABELS[m] for m in METHODS],
               loc="outside upper center", ncols=3, frameon=False,
               columnspacing=.8, handlelength=1.2, handletextpad=.4)
    legend_columns = []
    for parameterization, marker, style, label in (("blocked", "o", "-", "Six blocks (solid)"),
                                                  ("full", "s", "--", "Full controls (dashed)")):
        entries = [(Line2D([],[],color=".2",ls=style,marker=marker,ms=3.5), label)]
        if any(r["parameterization"] == parameterization and not r["fully_eligible"]
               and r["median_seconds"] is not None for r in rows):
            crossed = (Line2D([],[],color=".2",ls="none",marker=marker,ms=6,markerfacecolor="white"),
                       Line2D([],[],color=".2",ls="none",marker="x",ms=4))
            entries.append((crossed, "Rejected six blocks" if parameterization == "blocked" else "Rejected full"))
        legend_columns.append(entries)
    if len(legend_columns[0]) != len(legend_columns[1]):
        min(legend_columns, key=len).append((Line2D([],[],alpha=0), ""))
    entries = [entry for column in legend_columns for entry in column]
    fig.legend([handle for handle, _ in entries], [label for _, label in entries],
               loc="outside lower center", ncols=2, frameon=False, columnspacing=1., handlelength=2.,
               handler_map={tuple:HandlerTuple(ndivide=1, pad=0)})
    outputs = []
    for extension in ("pdf", "png"):
        path = args.source/f"scalability_current.{extension}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
    lines = ["# Current-controller computational study", "", manifest["protocol"]["timing"], "",
             manifest["protocol"]["initialization"], "", manifest["protocol"]["blocking"], "",
             manifest["protocol"]["quality"], "", manifest["protocol"]["scope"], "",
             f"Totals: {totals['complete_cases']}/{totals['cases']} workers complete; {totals['timed_samples']} timed calls, "
             f"{totals['cooperative_accepted']}/{totals['cooperative_timed']} cooperative calls pass the local equilibrium checks; "
             f"{totals['adversarial_feasible']}/{totals['adversarial_timed']} adversarial calls are model-feasible. "
             f"Independent audit checks {totals['independently_checked']} warmup/timed records and {totals['matched_pairs']} full/blocked input pairs.", "",
             "| Method | N | H | Controls | Variables | Inequalities | Median all [s] | Feasible | Eligible | Status |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    number = lambda value: "--" if value is None else f"{value:.6g}"
    for r in rows:
        lines.append(f"| {r['method']} | {r['agents']} | {r['horizon']} | {r['parameterization']} | {r['variables']} | {r['inequalities']} | "
                     f"{number(r['median_seconds'])} | {r['feasible']}/{r['timed_samples']} | {r['eligible']}/{r['timed_samples']} | {r['status']} |")
    lines += ["", "Eligible means independent local equilibrium acceptance for cooperative modes, model feasibility only for adversarial BR.", "",
              "## Identical-model full/blocked ablation", "",
              "Blocking restricts the admissible control space. Ratios are reported only when both cases finish all repeats with their stated quality checks.", "",
              "| Method | N | H | Full median [s] | Blocked median [s] | Eligible runtime ratio |",
              "|---|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f"| {p['method']} | {p['agents']} | {p['horizon']} | {number(p['full']['median_seconds'])} | {number(p['blocked']['median_seconds'])} | {number(p['qualified_runtime_ratio'])} |")
    lines += ["", "![Scaling](scalability_current.png)", "",
              "Lines connect returned-attempt median runtimes, including rejected/incomplete cases. Filled markers require all five calls to pass the stated eligibility check; crossed hollow markers retain rejected/incomplete points with their method color and control shape. Error bars show min/max repeats. A 100 ms line is a workload reference, not an end-to-end deadline certificate."]
    (args.source/"report.md").write_text("\n".join(lines)+"\n")
    renderer = args.source/"report_sources"/Path(__file__).name
    renderer.parent.mkdir(parents=True, exist_ok=True)
    renderer.write_bytes(Path(__file__).read_bytes())
    inputs = {os.path.relpath(args.source/name, ROOT): digest(args.source/name) for name in ("manifest.json", "summary.json", "validation.json", "report_data.json")}
    artifact = dict(command=[sys.executable,str(Path(__file__).relative_to(ROOT)),*sys.argv[1:]],
                    renderer_sha256=digest(Path(__file__)), input_sha256=inputs,
                    output_sha256={p.name:digest(p) for p in outputs}, totals=totals,
                    layout="Single column: side-by-side sweeps, shared logarithmic time axis, full scenario labels and explicit control/line-style legend; captions below each panel.",
                    connected_runtimes="Lines connect returned-attempt medians regardless of acceptance; filled versus crossed hollow markers encode check outcomes.",
                    rejected_markers="Crossed hollow circles/squares retain method colors and blocked/full parameterization; adjacent labels identify the method.",
                    figure_inches=fig.get_size_inches().tolist())
    (args.source/"artifact_manifest.json").write_text(json.dumps(artifact,indent=2)+"\n")
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        for path in outputs:
            shutil.copyfile(path,args.figures/path.name)
        (args.figures/"scalability_current_manifest.json").write_text(json.dumps(artifact,indent=2)+"\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()


def render_compact(rows, output):
    """Render stored numeric timing/eligibility rows without claiming a new audit."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":7.5, "axes.labelsize":7.5, "axes.titlesize":7.5,
                         "xtick.labelsize":7, "ytick.labelsize":7, "legend.fontsize":7, "pdf.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8), sharey=True, layout="constrained")
    for method in METHODS:
        curves(axes[0], rows, method, "blocked", "agents")
        for parameterization in ("blocked", "full"):
            curves(axes[1], rows, method, parameterization, "horizon")
    axes[0].set(xlabel="Agents $N$", xticks=[2,4,6,8,10], ylabel="Complete solve [s]")
    axes[1].set(xlabel="Horizon $H$", xticks=[10,20,30,40])
    for axis, caption in zip(axes, ("(a) $H=20$", "(b) $N=3/5$")):
        axis.annotate(caption, xy=(.5, 0), xycoords="axes fraction", xytext=(0, -34),
                      textcoords="offset points", ha="center", va="top", fontsize=7.5,
                      annotation_clip=False)
    for axis in axes:
        axis.set(yscale="log")
        axis.grid(alpha=.2, which="both")
        axis.axhline(.1, color=".45", lw=.8, ls=":")
    fig.legend([Line2D([],[],color=COLORS[m],lw=1.3) for m in METHODS],
               [LABELS[m] for m in METHODS],
               loc="outside upper center", ncols=3, frameon=False,
               columnspacing=.8, handlelength=1.2, handletextpad=.4)
    legend_columns = []
    for parameterization, marker, style, label in (("blocked", "o", "-", "Six blocks (solid)"),
                                                  ("full", "s", "--", "Full controls (dashed)")):
        entries = [(Line2D([],[],color=".2",ls=style,marker=marker,ms=3.5), label)]
        if any(r["parameterization"] == parameterization and not r["fully_eligible"]
               and r["median_seconds"] is not None for r in rows):
            crossed = (Line2D([],[],color=".2",ls="none",marker=marker,ms=6,markerfacecolor="white"),
                       Line2D([],[],color=".2",ls="none",marker="x",ms=4))
            entries.append((crossed, "Rejected six blocks" if parameterization == "blocked" else "Rejected full"))
        legend_columns.append(entries)
    if len(legend_columns[0]) != len(legend_columns[1]):
        min(legend_columns, key=len).append((Line2D([],[],alpha=0), ""))
    entries = [entry for column in legend_columns for entry in column]
    fig.legend([handle for handle, _ in entries], [label for _, label in entries],
               loc="outside lower center", ncols=2, frameon=False, columnspacing=1., handlelength=2.,
               handler_map={tuple:HandlerTuple(ndivide=1, pad=0)})
    outputs = []
    for extension in ("pdf", "png"):
        path = output/f"scalability_current.{extension}"
        fig.savefig(path, dpi=200)
        outputs.append(path)
    plt.close(fig)
