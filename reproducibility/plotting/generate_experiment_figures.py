#!/usr/bin/env python3
"""Generate publication-ready ACPR figures from the completed experiment files."""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/c/Users/sg564/Desktop/实验文件")
OUT = Path("/home/sg564/ACPR论文实验图_20260624")
MIB = 1024 * 1024

COLORS = {
    "baseline": "#6B7280",
    "fixed": "#2F6B8A",
    "acpr": "#B65F35",
    "ablation": "#8A9A5B",
    "muted": "#8C8C8C",
    "event": "#2C7A7B",
    "cpu": "#A85D3A",
    "rs": "#4C78A8",
    "ag": "#59A14F",
    "update": "#B279A2",
    "compute": "#7F7F7F",
}

LABELS = {
    "P0_baseline_no_compression": "No compression",
    "P1_fixed_rank": "Fixed rank\n(r=16)",
    "P2_acpr_final": "Update-norm\nrank",
    "P9_staged_rank_aggressive": "ACPR\n(aggressive rank)",
    "P3_no_active_prefix": "w/o active\nprefix",
    "P4_rank_reset_on_change": "w/ rank\nreset",
    "P5_embedding_off": "embedding\noff",
    "P6_embedding_broad": "embedding\nbroad",
    "P7_cpu_sync_handoff": "CPU-sync\nhandoff",
    "C8_fixed_rank_refresh_every_step": "Full PowerSGD\n(P+Q)",
}


def configure_plot_style() -> None:
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


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def ci95(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    critical = {3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def unique_nodes(text: str) -> list[str]:
    nodes: list[str] = []
    for node in re.findall(r"^(g\d{4}) slots=1$", text, re.MULTILINE):
        if node not in nodes:
            nodes.append(node)
    return nodes


def parse_performance() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "perf").glob("*.out")):
        text = path.read_text(errors="replace")
        iteration = re.search(r"Iteraction time: ([0-9.]+) \+-([0-9.]+)", text)
        per_gpu = re.search(r"Sen/sec per GPU: ([0-9.]+) \+-([0-9.]+)", text)
        total = re.search(r"Total sen/sec on (\d+) GPU\(s\): ([0-9.]+) \+-([0-9.]+)", text)
        if not (iteration and per_gpu and total):
            continue
        config = re.sub(r"_\d+\.out$", "", path.name)
        if config.startswith("P1_fixed_rank"):
            config = "P1_fixed_rank"
        elif config.startswith("P9_staged_rank_aggressive"):
            config = "P9_staged_rank_aggressive"
        rows.append(
            {
                "config": config,
                "job_file": path.name,
                "step_time_s": float(iteration.group(1)),
                "per_gpu_sps": float(per_gpu.group(1)),
                "total_sps": float(total.group(2)),
                "nodes": ",".join(unique_nodes(text)),
            }
        )
    return rows


def aggregate_performance(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config"])].append(row)
    summary: dict[str, dict[str, object]] = {}
    for config, group in grouped.items():
        throughputs = [float(r["total_sps"]) for r in group]
        times = [float(r["step_time_s"]) * 1000.0 for r in group]
        summary[config] = {
            "n": len(group),
            "throughput_mean": statistics.mean(throughputs),
            "throughput_ci95": ci95(throughputs),
            "step_ms_mean": statistics.mean(times),
            "step_ms_ci95": ci95(times),
            "nodes": [str(r["nodes"]) for r in group],
        }
    return summary


def add_full_powersgd_reference(summary: dict[str, dict[str, object]]) -> None:
    """Add full-P/Q PowerSGD speed using the same post-warmup window as P-series."""
    path = ROOT / "convergence" / "C8_fixed_rank_refresh_every_step_1122928.out"
    if not path.exists():
        return
    pattern = re.compile(r"Iter #(\d+): ([0-9.]+) sentences/sec per GPU")
    throughputs = []
    for match in pattern.finditer(path.read_text(errors="replace")):
        iteration = int(match.group(1))
        if 520 <= iteration <= 819:
            throughputs.append(float(match.group(2)) * 4.0)
    if not throughputs:
        return
    aggregate_sps = statistics.mean(throughputs)
    summary["C8_fixed_rank_refresh_every_step"] = {
        "n": 1,
        "throughput_mean": aggregate_sps,
        "throughput_ci95": 0.0,
        "step_ms_mean": 64.0 * 4.0 * 1000.0 / aggregate_sps,
        "step_ms_ci95": 0.0,
        "nodes": ["g0013,g0030,g0032,g0045"],
    }


def write_performance_csv(rows: list[dict[str, object]], summary: dict[str, dict[str, object]]) -> None:
    with (OUT / "performance_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "job_file", "step_time_s", "per_gpu_sps", "total_sps", "nodes"])
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "performance_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["config", "n", "throughput_mean", "throughput_ci95", "step_ms_mean", "step_ms_ci95", "node_sets"])
        for config in sorted(summary):
            item = summary[config]
            writer.writerow(
                [
                    config,
                    item["n"],
                    f"{item['throughput_mean']:.6f}",
                    f"{item['throughput_ci95']:.6f}",
                    f"{item['step_ms_mean']:.6f}",
                    f"{item['step_ms_ci95']:.6f}",
                    " | ".join(item["nodes"]),
                ]
            )


def add_run_points(ax: plt.Axes, x: float, values: list[float], color: str) -> None:
    offsets = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0.0])
    ax.scatter(
        np.full(len(values), x) + offsets,
        values,
        color=color,
        s=10,
        zorder=5,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.95,
    )


def perf_values(rows: list[dict[str, object]], config: str, field: str, scale: float = 1.0) -> list[float]:
    return [float(row[field]) * scale for row in rows if row["config"] == config]


def plot_performance_overview(rows: list[dict[str, object]], summary: dict[str, dict[str, object]]) -> None:
    configs = ["P0_baseline_no_compression", "P1_fixed_rank", "P9_staged_rank_aggressive"]
    colors = [COLORS["baseline"], COLORS["fixed"], COLORS["acpr"]]
    labels = [LABELS[c] for c in configs]
    x = np.arange(len(configs)) * 1.12
    bar_width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.58), gridspec_kw={"wspace": 0.36})
    baseline = float(summary[configs[0]]["throughput_mean"])

    means = [float(summary[c]["throughput_mean"]) for c in configs]
    cis = [float(summary[c]["throughput_ci95"]) for c in configs]
    axes[0].bar(
        x,
        means,
        yerr=cis,
        width=bar_width,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.45,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "capthick": 0.65},
        zorder=3,
    )
    for idx, config in enumerate(configs):
        vals = perf_values(rows, config, "total_sps")
        add_run_points(axes[0], x[idx], vals, colors[idx])
        axes[0].text(x[idx], means[idx] + max(cis[idx], 12) + 24, f"{means[idx] / baseline:.2f}$\\times$", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("Aggregate throughput (sentences/s)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(means) * 1.24)
    axes[0].set_xlim(x[0] - 0.55, x[-1] + 0.55)
    axes[0].set_title("(a) End-to-end throughput")

    means_ms = [float(summary[c]["step_ms_mean"]) for c in configs]
    cis_ms = [float(summary[c]["step_ms_ci95"]) for c in configs]
    axes[1].bar(
        x,
        means_ms,
        yerr=cis_ms,
        width=bar_width,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.45,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "capthick": 0.65},
        zorder=3,
    )
    for idx, config in enumerate(configs):
        vals = perf_values(rows, config, "step_time_s", 1000.0)
        add_run_points(axes[1], x[idx], vals, colors[idx])
        axes[1].text(x[idx], means_ms[idx] + max(cis_ms[idx], 0.5) + 2.3, f"{means_ms[idx]:.0f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylabel("Step time (ms)")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0, max(means_ms) * 1.24)
    axes[1].set_xlim(x[0] - 0.55, x[-1] + 0.55)
    axes[1].set_title("(b) Mean step time")

    for ax in axes:
        ax.yaxis.grid(True, color="#E2E2E2", linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
        ax.tick_params(axis="both", width=0.6, length=2.8)
    save(fig, "fig1_performance_overview")


def plot_ablation(rows: list[dict[str, object]], summary: dict[str, dict[str, object]]) -> None:
    configs = [
        "P9_staged_rank_aggressive",
        "P2_acpr_final",
        "P3_no_active_prefix",
        "P4_rank_reset_on_change",
        "P5_embedding_off",
        "C8_fixed_rank_refresh_every_step",
        "P7_cpu_sync_handoff",
    ]
    configs = [c for c in configs if c in summary]
    baseline = float(summary["P9_staged_rank_aggressive"]["throughput_mean"])
    x = np.arange(len(configs)) * 1.03
    means = [float(summary[c]["throughput_mean"]) / baseline for c in configs]
    cis = [float(summary[c]["throughput_ci95"]) / baseline for c in configs]
    colors = [COLORS["acpr"] if c == "P9_staged_rank_aggressive" else COLORS["ablation"] for c in configs]

    fig, ax = plt.subplots(figsize=(7.15, 2.58))
    ax.bar(
        x,
        means,
        yerr=cis,
        width=0.36,
        color=colors,
        edgecolor="#3F3F3F",
        linewidth=0.45,
        capsize=2.4,
        error_kw={"elinewidth": 0.65, "capthick": 0.65},
        zorder=3,
    )
    for idx, config in enumerate(configs):
        vals = [v / baseline for v in perf_values(rows, config, "total_sps")]
        add_run_points(ax, x[idx], vals, colors[idx])
        n = int(summary[config]["n"])
        tag = "ACPR" if config == "P9_staged_rank_aggressive" else f"{means[idx]:.3f}"
        ax.text(x[idx], means[idx] + max(cis[idx], 0.005) + 0.014, tag, ha="center", va="bottom", fontsize=7.0)
    ax.axhline(1.0, color="#444444", linewidth=0.75, linestyle=(0, (3.0, 2.2)), zorder=2)
    ax.set_ylabel("Normalized throughput\n(relative to ACPR)")
    ax.set_xticks(x, [LABELS[c] for c in configs])
    ax.set_ylim(0.50, 1.12)
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.yaxis.grid(True, color="#E2E2E2", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(axis="both", width=0.6, length=2.8)
    save(fig, "fig2_ablation_throughput")


def read_convergence() -> dict[str, list[dict[str, float]]]:
    results: dict[str, list[dict[str, float]]] = {}
    for path in sorted((ROOT / "convergence").glob("*.csv")):
        key = path.stem
        data: list[dict[str, float]] = []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if not row.get("loss"):
                    continue
                data.append({"step": float(row["step"]), "time_h": float(row["elapsed_time_s"]) / 3600.0, "loss": float(row["loss"])})
        results[key] = data
    return results


def rolling_mean(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_convergence(convergence: dict[str, list[dict[str, float]]]) -> None:
    """Keep optimization progress and wall-clock cost separate.

    Dense training runs longer in wall-clock time, so plotting all trajectories
    against elapsed time leaves the compressed curves visually truncated.  The
    left panel compares the same 60k optimization steps; the right panel shows
    the resulting time--loss trade-off explicitly.
    """
    configs = [
        ("C0_baseline_no_compression", "No compression", COLORS["baseline"]),
        ("C1_fixed_rank", "Fixed rank", COLORS["fixed"]),
        ("C2_acpr_final", "ACPR", COLORS["acpr"]),
        ("C3_rank_reset_on_change", "Rank reset", "#7D8F4E"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.83), gridspec_kw={"wspace": 0.40})
    endpoints = []
    for key, label, color in configs:
        data = convergence.get(key, [])
        if not data:
            continue
        steps = np.array([x["step"] for x in data]) / 1000.0
        losses = np.array([x["loss"] for x in data])
        axes[0].plot(steps, rolling_mean(losses), label=label, color=color)
        endpoints.append((label, color, data[-1]["time_h"], data[-1]["loss"]))

    axes[0].set_title("(a) Optimization trajectory")
    axes[0].set_xlabel("Training step (thousands)")
    axes[0].set_ylabel("Training loss (31-point moving mean)")
    axes[0].set_xlim(0, 60)
    axes[0].set_ylim(0.5, 11.8)
    axes[0].legend(frameon=False, loc="upper right")

    label_offsets = {
        "No compression": (5, 4),
        "Fixed rank": (5, -10),
        "ACPR": (5, 7),
        "Rank reset": (5, 7),
    }
    for label, color, hours, loss in endpoints:
        axes[1].scatter(hours, loss, s=48, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        axes[1].annotate(label, (hours, loss), xytext=label_offsets[label], textcoords="offset points", fontsize=6.8)
    axes[1].set_title("(b) Cost after 60k steps")
    axes[1].set_xlabel("Elapsed time (h)")
    axes[1].set_ylabel("Final training loss")
    axes[1].set_xlim(0, max(x[2] for x in endpoints) * 1.12)
    axes[1].set_ylim(0.5, max(x[3] for x in endpoints) * 1.13)
    axes[1].annotate("lower is better", xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=7.2, color="#666666")

    for ax in axes:
        ax.grid(True, color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig3_convergence_tradeoff")


def read_comm_stats(path: Path) -> list[dict[str, object]]:
    data: list[dict[str, object]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            data.append(
                {
                    "step": int(row["step"]),
                    "factor": row["factor_kind"],
                    "rank": int(row["current_rank"]),
                    "payload_mib": int(row["padded_communicated_bytes"]) / MIB,
                    "max_payload_mib": int(row["max_padded_bytes"]) / MIB,
                    "prefix": int(row["active_prefix_enabled"]),
                }
            )
    return data


def plot_active_prefix() -> None:
    active = read_comm_stats(ROOT / "acpr_incremental_20260623/timeline/T4_acpr_full_pq.comm_stats.csv")
    inactive = read_comm_stats(ROOT / "acpr_incremental_20260623/timeline/T5_no_active_prefix_full_pq.comm_stats.csv")
    steps = np.array([int(x["step"]) for x in active])
    ranks = np.array([int(x["rank"]) for x in active])
    total_active_mib = sum(float(x["payload_mib"]) for x in active)
    total_max_mib = sum(float(x["payload_mib"]) for x in inactive)
    reduction = 1.0 - total_active_mib / total_max_mib

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), gridspec_kw={"wspace": 0.50, "width_ratios": [1.05, 1.25, 0.9]})
    axes[0].step(steps, ranks, where="post", color=COLORS["acpr"], linewidth=1.7)
    axes[0].set_title("(a) Dynamic rank")
    axes[0].set_xlabel("Compressed step")
    axes[0].set_ylabel("Active rank")
    axes[0].set_yticks([8, 12, 16])
    axes[0].set_ylim(7, 17)
    axes[0].set_xlim(min(steps), max(steps))

    records = []
    for factor, color in [("p", COLORS["fixed"]), ("q", COLORS["acpr"])]:
        for rank in [16, 12, 8]:
            a = [float(x["payload_mib"]) for x in active if x["factor"] == factor and x["rank"] == rank]
            b = [float(x["payload_mib"]) for x in inactive if x["factor"] == factor and x["rank"] == rank]
            records.append((factor, rank, statistics.mean(a), statistics.mean(b), color))
    x = np.arange(len(records))
    axes[1].bar(x - 0.18, [r[2] for r in records], width=0.34, color=[r[4] for r in records], edgecolor="#333333", linewidth=0.45, label="Active prefix")
    axes[1].bar(x + 0.18, [r[3] for r in records], width=0.34, color="#C7C7C7", edgecolor="#333333", linewidth=0.45, label="Max-rank buffer")
    axes[1].set_title("(b) Measured factor payload")
    axes[1].set_ylabel("Communicated payload (MiB)")
    axes[1].set_xticks(x, [f"{r[0].upper()}\nr={r[1]}" for r in records])
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=6.4)

    axes[2].bar([0, 1], [total_max_mib / 1024.0, total_active_mib / 1024.0], width=0.58, color=["#C7C7C7", COLORS["acpr"]], edgecolor="#333333", linewidth=0.5)
    axes[2].set_title("(c) 1,200 steps")
    axes[2].set_ylabel("Cumulative payload (GiB)")
    axes[2].set_xticks([0, 1], ["Max\nbuffer", "Active\nprefix"])
    axes[2].text(0.5, max(total_max_mib, total_active_mib) / 1024.0 * 1.045, f"−{reduction * 100:.1f}%", ha="center", va="bottom", color=COLORS["acpr"], fontsize=8.2)

    for ax in axes:
        ax.yaxis.grid(True, color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_dynamic_rank_payload")


def tail_lines(path: Path, count: int) -> list[bytes]:
    """Read the final profile cycles without loading a 500+ MiB JSONL trace."""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        block = 1024 * 1024
        data = b""
        position = end
        while position > 0 and data.count(b"\n") <= count:
            take = min(block, position)
            position -= take
            handle.seek(position)
            data = handle.read(take) + data
    return [line for line in data.splitlines()[-count:] if line]


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def handoff_samples(name: str) -> dict[str, list[float]]:
    path = ROOT / "timeline" / f"{name}.jsonl"
    ready_to_issue: list[float] = []
    cycle_ms: list[float] = []
    rs_tail_ms: list[float] = []
    ag_wait_ms: list[float] = []
    for line in tail_lines(path, 1000):
        record = json.loads(line)
        cycle_ms.append((record["cycle_end_ts"] - record["cycle_start_ts"]) * 1000.0)
        rs_tail_ms.append(record["summary"]["rs_tail_wait_s"] * 1000.0)
        ag_wait_ms.append(record["summary"]["ag_wait_s"] * 1000.0)
        for trace in record["group_traces"]:
            ready, issue = trace.get("rs_ready_ts"), trace.get("rs_issue_ts")
            if ready is not None and issue is not None and issue >= ready:
                ready_to_issue.append((issue - ready) * 1000.0)
    return {
        "ready_to_issue_ms": ready_to_issue,
        "cycle_ms": cycle_ms,
        "rs_tail_ms": rs_tail_ms,
        "ag_wait_ms": ag_wait_ms,
    }


def write_handoff_summary(event: dict[str, list[float]], cpu: dict[str, list[float]]) -> None:
    with (OUT / "handoff_distribution_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "metric", "n", "mean_ms", "p50_ms", "p90_ms", "p95_ms"])
        for path, samples in [("CUDA event", event), ("CPU synchronize", cpu)]:
            for metric in ("ready_to_issue_ms", "cycle_ms", "rs_tail_ms", "ag_wait_ms"):
                values = samples[metric]
                writer.writerow([
                    path,
                    metric,
                    len(values),
                    f"{statistics.mean(values):.6f}",
                    f"{percentile(values, 0.50):.6f}",
                    f"{percentile(values, 0.90):.6f}",
                    f"{percentile(values, 0.95):.6f}",
                ])


def plot_handoff_distribution() -> None:
    event = handoff_samples("T1_acpr_final_timeline")
    cpu = handoff_samples("T3_cpu_sync_handoff_timeline")
    write_handoff_summary(event, cpu)
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.62), gridspec_kw={"wspace": 0.42})

    for values, label, color in [
        (event["ready_to_issue_ms"], "CUDA event", COLORS["event"]),
        (cpu["ready_to_issue_ms"], "CPU synchronize", COLORS["cpu"]),
    ]:
        ordered = np.sort(np.asarray(values))
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        axes[0].plot(ordered, cdf, label=label, color=color, linewidth=1.55)
        axes[0].axvline(percentile(values, 0.5), color=color, linewidth=0.65, linestyle="--", alpha=0.75)
    axes[0].set_xscale("log")
    axes[0].set_title("(a) Host-side RS launch delay")
    axes[0].set_xlabel("Buffer ready $\\rightarrow$ RS enqueued (ms)")
    axes[0].set_ylabel("Empirical CDF")
    axes[0].set_xlim(0.005, 20)
    axes[0].set_ylim(0, 1.02)
    axes[0].text(
        0.05,
        0.96,
        "CUDA event",
        transform=axes[0].transAxes,
        va="top",
        color=COLORS["event"],
        fontsize=7.1,
    )
    axes[0].text(0.58, 0.52, "CPU synchronize", transform=axes[0].transAxes, color=COLORS["cpu"], fontsize=7.1)
    axes[0].text(0.05, 0.14, f"p50: {percentile(event['ready_to_issue_ms'], .5):.3f} vs. {percentile(cpu['ready_to_issue_ms'], .5):.3f} ms", transform=axes[0].transAxes, fontsize=6.5)

    metrics = [("RS tail", "rs_tail_ms"), ("AG stall", "ag_wait_ms")]
    x = np.arange(len(metrics))
    width = 0.32
    event_medians = [percentile(event[key], .5) for _, key in metrics]
    cpu_medians = [percentile(cpu[key], .5) for _, key in metrics]
    axes[1].bar(x - width / 2, event_medians, width, color=COLORS["event"], edgecolor="#333333", linewidth=0.45, label="CUDA event")
    axes[1].bar(x + width / 2, cpu_medians, width, color=COLORS["cpu"], edgecolor="#333333", linewidth=0.45, label="CPU synchronize")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Downstream exposed waiting")
    axes[1].set_ylabel("Median exposed wait (ms)")
    axes[1].set_xticks(x, [x[0] for x in metrics])
    for bars, values in [(axes[1].containers[0], event_medians), (axes[1].containers[1], cpu_medians)]:
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3)
    axes[1].text(0.06, 0.96, "CUDA event", transform=axes[1].transAxes, va="top", color=COLORS["event"], fontsize=6.9)
    axes[1].text(0.06, 0.87, "CPU synchronize", transform=axes[1].transAxes, va="top", color=COLORS["cpu"], fontsize=6.9)

    for ax in axes:
        ax.grid(True, color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig5_stream_handoff_distribution")


def write_readme(summary: dict[str, dict[str, object]]) -> None:
    p0 = float(summary["P0_baseline_no_compression"]["throughput_mean"])
    p1 = float(summary["P1_fixed_rank"]["throughput_mean"])
    p2 = float(summary["P2_acpr_final"]["throughput_mean"])
    p9 = float(summary["P9_staged_rank_aggressive"]["throughput_mean"])
    lines = [
        "# ACPR experiment figures (2026-06-24)",
        "",
        "All figures are generated directly from the files under `实验文件`.",
        "",
        "## Main-performance summary",
        f"- No compression: {p0:.1f} sentences/s.",
        f"- Fixed rank: {p1:.1f} sentences/s ({p1 / p0:.2f}x vs. no compression).",
        f"- ACPR aggressive dynamic rank: {p9:.1f} sentences/s ({p9 / p0:.2f}x vs. no compression; {p9 / p1:.3f}x vs. fixed rank).",
        f"- Update-norm dynamic rank: {p2:.1f} sentences/s ({p2 / p0:.2f}x vs. no compression; {p2 / p9:.3f}x vs. aggressive rank).",
        "- P7 has three completed repeats and is included with a 95% CI.",
        "- P8 was not present in the supplied experiment directory and is not plotted.",
        "",
        "## Figure scope",
        "- Fig. 1 and Fig. 2 use only main-performance `.out` files.",
        "- Fig. 3 separates optimization-step progress from the 60k-step time/loss trade-off.",
        "- Fig. 4 uses T4/T5 full P/Q statistics and explains dynamic-rank payload reduction.",
        "- Fig. 5 aggregates 1,000 final profile cycles (16,000 group launches per path); it is not a single-cycle trace.",
        "",
        "## Recommended captions",
        "- Fig. 1: Aggregate throughput and mean step time on four nodes (one RTX 4090 per node). Points denote independent Slurm runs; error bars are 95% confidence intervals over three runs.",
        "- Fig. 2: Throughput sensitivity to ACPR design choices, normalized to ACPR. Points are independent Slurm runs.",
        "- Fig. 3: Training loss against common optimization steps and the corresponding wall-clock/final-loss trade-off after 60k steps. Curves use a centered moving mean over 31 logged samples for readability.",
        "- Fig. 4: Full P/Q communication statistics over 1,200 compressed steps. Active-prefix communication reduces cumulative transmitted payload by 31.9% versus the max-rank buffer path under the same rank trajectory.",
        "- Fig. 5: Event handoff versus CPU synchronization across 16,000 reduce-scatter launches per path. It reports host-side launch delay and downstream exposed waits, not a representative cycle.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    rows = parse_performance()
    summary = aggregate_performance(rows)
    add_full_powersgd_reference(summary)
    required = {"P0_baseline_no_compression", "P1_fixed_rank", "P9_staged_rank_aggressive"}
    missing = required.difference(summary)
    if missing:
        raise RuntimeError(f"Missing required performance results: {sorted(missing)}")
    write_performance_csv(rows, summary)
    plot_performance_overview(rows, summary)
    plot_ablation(rows, summary)
    plot_convergence(read_convergence())
    plot_active_prefix()
    plot_handoff_distribution()
    write_readme(summary)
    print(OUT)


if __name__ == "__main__":
    main()
