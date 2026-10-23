#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

BASE = Path('/home/sg564/acpr_cifar_analysis_20260629')
OUT_PDF = BASE / 'fig_eval_convergence_cifar_accuracy.pdf'
OUT_PNG = BASE / 'fig_eval_convergence_cifar_accuracy.png'
OUT_CSV = BASE / 'cifar_accuracy_summary.csv'

COLORS = {
    'baseline': '#6B7280',
    'fixed': '#2F6B8A',
    'acpr': '#B65F35',
    'full': '#8A9A5B',
    'grid': '#D9D9D9',
    'text': '#333333',
}

METHODS = [
    ('Dense DeAR', 'CF0_cifar100_resnet34_dense.csv', COLORS['baseline'], 'Dense'),
    ('Fixed rank', 'CF1_cifar100_resnet34_fixed_rank.csv', COLORS['fixed'], 'Fixed\nrank'),
    ('ACPR', 'CF2_cifar100_resnet34_acpr_aggressive.csv', COLORS['acpr'], 'ACPR'),
    ('Full PowerSGD', 'CF3_cifar100_resnet34_full_powersgd.csv', COLORS['full'], 'Full\nPowerSGD'),
]


def configure_plot_style() -> None:
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 7.2,
        'axes.labelsize': 8.0,
        'axes.titlesize': 8.7,
        'xtick.labelsize': 7.1,
        'ytick.labelsize': 7.1,
        'legend.fontsize': 7.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.75,
        'lines.linewidth': 1.45,
    })


def read_validation(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row['kind'] != 'validation':
                continue
            rows.append({
                'epoch': int(row['epoch']),
                'global_step': int(row['global_step']),
                'elapsed_time_s': float(row['elapsed_time_s']),
                'time_min': float(row['elapsed_time_s']) / 60.0,
                'time_h': float(row['elapsed_time_s']) / 3600.0,
                'loss': float(row['loss']),
                'top1': float(row['top1']),
                'samples_seen': int(row['samples_seen']),
            })
    return rows


def moving_average(values: list[float], window: int = 5) -> list[float]:
    half = window // 2
    smoothed = []
    for idx in range(len(values)):
        left = max(0, idx - half)
        right = min(len(values), idx + half + 1)
        smoothed.append(sum(values[left:right]) / (right - left))
    return smoothed


def add_endpoint_label(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.scatter([x], [y], s=20, color=color, edgecolor='white', linewidth=0.55, zorder=5)


def main() -> None:
    configure_plot_style()
    series = []
    summary = []
    for label, filename, color, short in METHODS:
        rows = read_validation(BASE / filename)
        smooth = moving_average([r['top1'] for r in rows], 5)
        for row, value in zip(rows, smooth):
            row['top1_smooth'] = value
        final = rows[-1]
        best = max(rows, key=lambda r: r['top1'])
        series.append((label, short, rows, color))
        summary.append({
            'method': label,
            'final_epoch': final['epoch'],
            'final_time_s': final['elapsed_time_s'],
            'final_time_min': final['time_min'],
            'final_time_h': final['time_h'],
            'final_val_loss': final['loss'],
            'final_top1': final['top1'],
            'best_top1': best['top1'],
            'best_epoch': best['epoch'],
            'best_time_min': best['time_min'],
        })

    with OUT_CSV.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.15, 2.75),
        gridspec_kw={'width_ratios': [1.54, 1.0], 'wspace': 0.35},
    )
    ax, ax_bar = axes

    for label, short, rows, color in series:
        epochs = [r['epoch'] + 1 for r in rows]
        top1 = [r['top1'] for r in rows]
        smooth = [r['top1_smooth'] for r in rows]
        ax.plot(epochs, smooth, color=color, linewidth=1.45, label=label)
        ax.plot(epochs[::30], top1[::30], linestyle='none', marker='o', ms=2.2, color=color, alpha=0.68, markeredgecolor='white', markeredgewidth=0.35)
        add_endpoint_label(ax, epochs[-1], smooth[-1], f'{top1[-1]:.2f}', color)

    ax.set_title('(a) Validation accuracy by epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Top-1 (%)')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 84)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.legend(frameon=False, loc='lower right', ncol=2, handlelength=1.8, columnspacing=1.0)

    labels = [item[3] for item in METHODS]
    x = np.arange(len(summary))
    final_top1 = [row['final_top1'] for row in summary]
    final_time = [row['final_time_min'] for row in summary]
    colors = [item[2] for item in METHODS]
    bars = ax_bar.bar(x, final_top1, width=0.62, color=colors, edgecolor='#333333', linewidth=0.55)
    ax_bar.axhline(final_top1[0], color=COLORS['baseline'], linewidth=0.8, linestyle='--', alpha=0.85)
    ax_bar.set_title('(b) Final validation accuracy')
    ax_bar.set_ylabel('Top-1 at epoch 300 (%)')
    ax_bar.set_xticks(x, labels)
    ax_bar.set_ylim(76.8, 78.8)
    ax_bar.yaxis.set_major_locator(MaxNLocator(5))
    for rect, acc in zip(bars, final_top1):
        ax_bar.text(rect.get_x() + rect.get_width() / 2, acc + 0.035, f'{acc:.2f}', ha='center', va='bottom', fontsize=7.3)

    for axis in axes:
        axis.yaxis.grid(True, color=COLORS['grid'], linewidth=0.55)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(width=0.65, length=2.8)

    fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(OUT_PDF)
    for row in summary:
        print('{method}: final={final_top1:.2f}, best={best_top1:.2f}@{best_epoch}, time={final_time_min:.2f}min, loss={final_val_loss:.4f}'.format(**row))


if __name__ == '__main__':
    main()
