#!/usr/bin/env python3
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/mnt/c/Users/sg564/Desktop/Helix技术论文/实验文件/Figures_20260624/formal_figures_p1p9_fullpowersgd_20260628")
OUT = Path(__file__).parent / "source/figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "baseline": "#6B7280",
    "fixed": "#2F6B8A",
    "helix": "#B65F35",
}

CONFIGS = [
    ("P0_baseline_no_compression", "No\ncompression", COLORS["baseline"]),
    ("P9_staged_rank_aggressive", "Helix\n(ours)", COLORS["helix"]),
]


def ci95(values):
    if len(values) < 3:
        return 0.0
    critical = {3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def load_runs():
    rows = []
    with (BASE / "performance_runs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    return grouped


def values(grouped, config, key, scale=1.0):
    return [float(row[key]) * scale for row in grouped[config]]


def add_run_points(ax, x, vals, color):
    offsets = np.linspace(-0.055, 0.055, len(vals)) if len(vals) > 1 else np.array([0.0])
    ax.scatter(
        np.full(len(vals), x) + offsets,
        vals,
        color=color,
        s=10,
        zorder=5,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
    )


def main():
    grouped = load_runs()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.45,
        }
    )

    x = np.arange(len(CONFIGS)) * 1.12
    bar_width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.58), gridspec_kw={"wspace": 0.36})

    throughput = [values(grouped, cfg, "total_sps") for cfg, _, _ in CONFIGS]
    means = [statistics.mean(v) for v in throughput]
    baseline = means[0]
    colors = [c for _, _, c in CONFIGS]
    labels = [label for _, label, _ in CONFIGS]

    axes[0].bar(
        x,
        means,
        width=bar_width,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.45,
        zorder=3,
    )
    for idx, vals in enumerate(throughput):
        add_run_points(axes[0], x[idx], vals, colors[idx])
        axes[0].text(
            x[idx],
            max(vals) + 36,
            f"{means[idx] / baseline:.2f}$\\times$",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    axes[0].set_ylabel("Aggregate throughput (sentences/s)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(means) * 1.24)
    axes[0].set_xlim(x[0] - 0.55, x[-1] + 0.55)
    axes[0].set_title("(a) Per-iteration throughput")

    step_ms = [values(grouped, cfg, "step_time_s", 1000.0) for cfg, _, _ in CONFIGS]
    means_ms = [statistics.mean(v) for v in step_ms]
    axes[1].bar(
        x,
        means_ms,
        width=bar_width,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.45,
        zorder=3,
    )
    for idx, vals in enumerate(step_ms):
        add_run_points(axes[1], x[idx], vals, colors[idx])
        axes[1].text(
            x[idx],
            max(vals) + 2.8,
            f"{means_ms[idx]:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    axes[1].set_ylabel("Step time (ms)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(means_ms) * 1.24)
    axes[1].set_xlim(x[0] - 0.55, x[-1] + 0.55)
    axes[1].set_title("(b) Mean step time")

    for ax in axes:
        ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(OUT / "fig_eval_performance.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(__file__).parent / "fig_eval_performance.png", dpi=200, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
