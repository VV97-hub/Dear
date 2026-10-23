#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path("/mnt/c/Users/sg564/Desktop/Helix技术论文/实验文件")
OUT = Path(__file__).parent / "source/figures"

PERF_RUNS = ROOT / "Figures_20260624/formal_figures_p1p9_fullpowersgd_20260628/performance_runs.csv"
PERF_SUMMARY = ROOT / "Figures_20260624/formal_figures_p1p9_fullpowersgd_20260628/performance_summary.csv"
CIFAR_DIR = ROOT / "Collected_20260706/cifar_convergence"

COLORS = {
    "baseline": "#737B88",
    "fixed": "#2F7591",
    "aldc": "#B86134",
    "ablation": "#8A9A5B",
    "full": "#8A9A5B",
    "grid": "#DDDDDD",
    "edge": "#333333",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 8.1,
            "axes.titlesize": 8.9,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 7.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.72,
            "lines.linewidth": 1.45,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.018)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["config"]] = {
                "throughput_mean": float(row["throughput_mean"]),
                "throughput_ci95": float(row["throughput_ci95"]),
                "step_ms_mean": float(row["step_ms_mean"]),
                "step_ms_ci95": float(row["step_ms_ci95"]),
                "n": float(row["n"]),
            }
    return out


def run_values(rows: list[dict[str, str]], config: str, field: str, scale: float = 1.0) -> list[float]:
    return [float(row[field]) * scale for row in rows if row["config"] == config]


def add_points(ax: plt.Axes, x: float, values: list[float], color: str) -> None:
    if not values:
        return
    offsets = np.linspace(-0.052, 0.052, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(
        np.full(len(values), x) + offsets,
        values,
        s=10.5,
        color=color,
        edgecolor="white",
        linewidth=0.38,
        zorder=5,
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=COLORS["grid"], linewidth=0.52)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.68)
    ax.spines["bottom"].set_linewidth(0.68)
    ax.tick_params(width=0.62, length=2.8)


def plot_fig7(rows: list[dict[str, str]], summary: dict[str, dict[str, float]]) -> None:
    configs = ["P0_baseline_no_compression", "P1_fixed_rank", "P9_staged_rank_aggressive"]
    labels = ["No\ncompression", "Fixed rank\n(r=16)", "Helix\n(ours)"]
    colors = [COLORS["baseline"], COLORS["fixed"], COLORS["aldc"]]
    x = np.arange(len(configs)) * 1.08
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.60), gridspec_kw={"wspace": 0.36})

    baseline = summary[configs[0]]["throughput_mean"]
    means = [summary[c]["throughput_mean"] for c in configs]
    cis = [summary[c]["throughput_ci95"] for c in configs]
    axes[0].bar(
        x,
        means,
        width=width,
        yerr=cis,
        color=colors,
        edgecolor=COLORS["edge"],
        linewidth=0.48,
        capsize=2.2,
        error_kw={"elinewidth": 0.65, "capthick": 0.65},
        zorder=3,
    )
    for idx, c in enumerate(configs):
        add_points(axes[0], x[idx], run_values(rows, c, "total_sps"), colors[idx])
        axes[0].text(
            x[idx],
            means[idx] + max(cis[idx], 12.0) + 23,
            f"{means[idx] / baseline:.2f}$\\times$",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    axes[0].set_title("(a) End-to-end throughput")
    axes[0].set_ylabel("Aggregate throughput (sentences/s)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(means) * 1.24)
    axes[0].set_xlim(x[0] - 0.50, x[-1] + 0.50)
    style_axis(axes[0])

    means_ms = [summary[c]["step_ms_mean"] for c in configs]
    cis_ms = [summary[c]["step_ms_ci95"] for c in configs]
    axes[1].bar(
        x,
        means_ms,
        width=width,
        yerr=cis_ms,
        color=colors,
        edgecolor=COLORS["edge"],
        linewidth=0.48,
        capsize=2.2,
        error_kw={"elinewidth": 0.65, "capthick": 0.65},
        zorder=3,
    )
    for idx, c in enumerate(configs):
        add_points(axes[1], x[idx], run_values(rows, c, "step_time_s", 1000.0), colors[idx])
        axes[1].text(
            x[idx],
            means_ms[idx] + max(cis_ms[idx], 0.8) + 2.0,
            f"{means_ms[idx]:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    axes[1].set_title("(b) Mean step time")
    axes[1].set_ylabel("Step time (ms)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(means_ms) * 1.24)
    axes[1].set_xlim(x[0] - 0.50, x[-1] + 0.50)
    style_axis(axes[1])
    save(fig, "fig_eval_performance")


def plot_fig8(rows: list[dict[str, str]], summary: dict[str, dict[str, float]]) -> None:
    configs = [
        "P9_staged_rank_aggressive",
        "P2_acpr_final",
        "P3_no_active_prefix",
        "P4_rank_reset_on_change",
        "P5_embedding_off",
        "C8_fixed_rank_refresh_every_step",
        "P7_cpu_sync_handoff",
    ]
    labels = [
        "Helix\n(ours)",
        "Update-norm\nrank",
        "w/o active\nprefix",
        "w/ rank\nreset",
        "Embedding\noff",
        "Full\nPowerSGD",
        "CPU-sync\nhandoff",
    ]
    baseline = summary["P9_staged_rank_aggressive"]["throughput_mean"]
    x = np.arange(len(configs)) * 1.03
    means = [summary[c]["throughput_mean"] / baseline for c in configs]
    cis = [summary[c]["throughput_ci95"] / baseline for c in configs]
    colors = [COLORS["aldc"]] + [COLORS["ablation"]] * (len(configs) - 1)

    fig, ax = plt.subplots(figsize=(7.15, 2.62))
    ax.bar(
        x,
        means,
        width=0.34,
        color=colors,
        edgecolor=COLORS["edge"],
        linewidth=0.48,
        zorder=3,
    )
    for idx, c in enumerate(configs):
        vals = [v / baseline for v in run_values(rows, c, "total_sps")]
        if not vals and c in summary:
            vals = [means[idx]]
        add_points(ax, x[idx], vals, colors[idx])
        ax.text(
            x[idx],
            means[idx] + max(cis[idx], 0.004) + 0.014,
            f"{means[idx]:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.1,
        )
    ax.axhline(1.0, color="#444444", linewidth=0.75, linestyle=(0, (3.0, 2.2)), zorder=2)
    ax.set_ylabel("Normalized throughput\n(relative to Helix)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.50, 1.12)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    style_axis(ax)
    save(fig, "fig_eval_ablation")


def read_validation(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kind"] != "validation":
                continue
            rows.append(
                {
                    "epoch": int(row["epoch"]) + 1,
                    "elapsed_time_s": float(row["elapsed_time_s"]),
                    "loss": float(row["loss"]),
                    "top1": float(row["top1"]),
                }
            )
    return rows


def moving_average(values: list[float], window: int = 5) -> list[float]:
    out: list[float] = []
    half = window // 2
    for idx in range(len(values)):
        left = max(0, idx - half)
        right = min(len(values), idx + half + 1)
        out.append(sum(values[left:right]) / (right - left))
    return out


def plot_fig9() -> None:
    methods = [
        ("Dense DeAR", "CF0_cifar100_resnet34_dense.csv", COLORS["baseline"], "Dense"),
        ("Fixed rank", "CF1_cifar100_resnet34_fixed_rank.csv", COLORS["fixed"], "Fixed\nrank"),
        ("Helix", "CF2_cifar100_resnet34_acpr_aggressive.csv", COLORS["aldc"], "Helix"),
        ("Full PowerSGD", "CF3_cifar100_resnet34_full_powersgd.csv", COLORS["full"], "Full\nPowerSGD"),
    ]
    series = []
    for label, file_name, color, short in methods:
        rows = read_validation(CIFAR_DIR / file_name)
        series.append((label, short, rows, color))

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"width_ratios": [1.54, 1.0], "wspace": 0.35})
    ax, ax_bar = axes
    for label, _short, rows, color in series:
        epochs = [r["epoch"] for r in rows]
        top1 = [r["top1"] for r in rows]
        smooth = moving_average(top1, 5)
        ax.plot(epochs, smooth, color=color, linewidth=1.48, label=label)
        ax.plot(
            epochs[::30],
            top1[::30],
            linestyle="none",
            marker="o",
            ms=2.1,
            color=color,
            alpha=0.68,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
    ax.set_title("(a) Validation accuracy by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Top-1 (%)")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 84)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.8, columnspacing=1.0)

    x = np.arange(len(series))
    labels = [item[1] for item in series]
    final_top1 = [item[2][-1]["top1"] for item in series]
    colors = [item[3] for item in series]
    bars = ax_bar.bar(x, final_top1, width=0.56, color=colors, edgecolor=COLORS["edge"], linewidth=0.52, zorder=3)
    ax_bar.axhline(final_top1[0], color=COLORS["baseline"], linewidth=0.8, linestyle="--", alpha=0.85)
    ax_bar.set_title("(b) Final validation accuracy")
    ax_bar.set_ylabel("Top-1 at epoch 300 (%)")
    ax_bar.set_xticks(x, labels)
    ax_bar.set_ylim(76.8, 78.8)
    ax_bar.yaxis.set_major_locator(MaxNLocator(5))
    for rect, acc in zip(bars, final_top1):
        ax_bar.text(rect.get_x() + rect.get_width() / 2, acc + 0.035, f"{acc:.2f}", ha="center", va="bottom", fontsize=7.3)

    for axis in axes:
        style_axis(axis)
    save(fig, "fig_eval_convergence")


def main() -> None:
    configure()
    rows = read_rows(PERF_RUNS)
    summary = read_summary(PERF_SUMMARY)
    plot_fig8(rows, summary)
    print(OUT)


if __name__ == "__main__":
    main()
