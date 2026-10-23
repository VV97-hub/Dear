from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch


OUT = Path("/home/sg564/acpr_mechanism_redraw_20260630")
OUT.mkdir(exist_ok=True)

for font_path in [
    "/mnt/c/Windows/Fonts/times.ttf",
    "/mnt/c/Windows/Fonts/timesbd.ttf",
    "/mnt/c/Windows/Fonts/timesi.ttf",
    "/mnt/c/Windows/Fonts/timesbi.ttf",
]:
    font_manager.fontManager.addfont(font_path)
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COLORS = {
    "compute": "#C7DDBB",
    "factor": "#F2C28A",
    "comm": "#A9C3E9",
    "active": "#7D96A3",
    "inactive": "#D7E0E3",
    "suffix": "#E3E0D7",
    "edge": "#5F6870",
    "text": "#22272E",
}


def add_box(ax, x, y, w, h, text, fc, ec=None, fs=8.5, weight="normal"):
    if ec is None:
        ec = COLORS["edge"]
    box = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=0.8)
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=COLORS["text"],
        weight=weight,
    )
    return box


def add_round(ax, x, y, w, h, text, fc, ec=None, fs=9, weight="normal"):
    if ec is None:
        ec = COLORS["edge"]
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=COLORS["text"],
        weight=weight,
    )
    return box


def add_arrow(ax, xy1, xy2, lw=0.8, color=None, style="-|>"):
    if color is None:
        color = COLORS["edge"]
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle=style,
            mutation_scale=8,
            linewidth=lw,
            color=color,
        )
    )


def draw_half_step():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.15), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.06, hspace=0.40)

    legend = fig.add_axes([0.23, 0.91, 0.58, 0.08])
    legend.axis("off")
    legend_xs = [0.00, 0.36, 0.70]
    for i, (label, color) in enumerate(
        [
            ("Gradient compute", COLORS["compute"]),
            ("Factor compute", COLORS["factor"]),
            ("RS / AG communication", COLORS["comm"]),
        ]
    ):
        x = legend_xs[i]
        legend.add_patch(Rectangle((x, 0.32), 0.065, 0.36, facecolor=color, edgecolor=COLORS["edge"], linewidth=0.8))
        legend.text(x + 0.08, 0.5, label, va="center", ha="left", fontsize=8.8)

    def panel(ax, title, half=False):
        ax.set_xlim(0, 12.2)
        ax.set_ylim(0, 4.55)
        ax.axis("off")
        ax.text(0.0, 4.38, title, fontsize=9.8, weight="bold", ha="left", va="top")

        y_time = 3.22
        ax.plot([0.45, 11.85], [y_time, y_time], color=COLORS["edge"], linewidth=0.8)
        add_arrow(ax, (11.55, y_time), (11.85, y_time), lw=0.8)
        ticks = [(0.95, "backward\nstart"), (2.05, "$G_2$ ready"), (4.05, "$G_1$ ready"), (6.05, "$G_0$ ready")]
        if half:
            # Align synchronization markers to the actual communication block
            # boundaries rather than to approximate phase markers.
            barrier_x = 6.05 + 0.48 + 0.75  # right edge of RS-F_0
            ag_start_x = barrier_x
            next_forward_x = ag_start_x + 3 * 0.70  # right edge of AG-F_2
            ticks += [(barrier_x, "backward\nend"), (next_forward_x, "next\nforward")]
        else:
            barrier_x = 7.25 + 0.42 + 0.80 + 0.42 + 0.80  # right edge of RS-Q_0
            ag_start_x = barrier_x
            next_forward_x = ag_start_x + 3 * 0.62  # right edge of AG-Q_2
            ticks += [(barrier_x, "backward\nend"), (next_forward_x, "next\nforward")]
        for x, lab in ticks:
            ax.plot([x, x], [y_time - 0.12, y_time + 0.12], color=COLORS["edge"], linewidth=0.7)
            ax.text(x, y_time + 0.18, lab, fontsize=7.2, ha="center", va="bottom")
        ax.text(12.12, y_time + 0.74, "forward", fontsize=7.5, ha="right")
        ax.text(0, y_time, "time", fontsize=7.8, ha="right", va="center")

        ax.text(0.1, 2.45, "Backward\ncompute", fontsize=7.8, ha="right", va="center")
        ax.text(0.1, 1.55, "Backward\ncomm.", fontsize=7.8, ha="right", va="center")
        ax.text(0.1, 0.62, "Forward\ngate", fontsize=7.8, ha="right", va="center")
        ax.plot([barrier_x, barrier_x], [0.35, 3.15], color=COLORS["edge"], linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(barrier_x + 0.04, 3.0, "barrier", fontsize=7.2, ha="left", va="center")

        add_box(ax, 0.9, 2.1, 1.15, 0.48, "$M_2$", COLORS["compute"], fs=10)
        add_box(ax, 2.05, 2.1, 2.0, 0.48, "$M_1$", COLORS["compute"], fs=10)
        add_box(ax, 4.05, 2.1, 2.0, 0.48, "$M_0$", COLORS["compute"], fs=10)

        if half:
            starts = [2.05, 4.05, 6.05]
            labels = ["2", "1", "0"]
            for x, lab in zip(starts, labels):
                ax.plot([x, x], [2.1, 1.85], color=COLORS["edge"], linewidth=0.7, linestyle=(0, (2, 2)))
                add_arrow(ax, (x, 1.85), (x, 1.72), lw=0.7)
                add_box(ax, x, 1.32, 0.48, 0.42, f"$F_{lab}$", COLORS["factor"], fs=8.2)
                add_box(ax, x + 0.48, 1.32, 0.75, 0.42, f"RS-$F_{lab}$", COLORS["comm"], fs=7.4)
            ax.text(7.62, 1.52, "$F \\in \\{P,Q\\}$", fontsize=8.2, ha="left", va="center")
            for i, lab in enumerate(["0", "1", "2"]):
                add_box(ax, ag_start_x + i * 0.70, 0.36, 0.70, 0.42, f"AG-$F_{lab}$", COLORS["comm"], fs=7.4)
        else:
            starts = [2.05, 4.65, 7.25]
            labels = ["2", "1", "0"]
            for x, lab in zip(starts, labels):
                ax.plot([x, x], [2.1, 1.85], color=COLORS["edge"], linewidth=0.7, linestyle=(0, (2, 2)))
                add_arrow(ax, (x, 1.85), (x, 1.72), lw=0.7)
                add_box(ax, x, 1.32, 0.42, 0.42, f"$P_{lab}$", COLORS["factor"], fs=8.0)
                add_box(ax, x + 0.42, 1.32, 0.80, 0.42, f"RS-$P_{lab}$", COLORS["comm"], fs=7.3)
                add_box(ax, x + 1.22, 1.32, 0.42, 0.42, f"$Q_{lab}$", COLORS["factor"], fs=8.0)
                add_box(ax, x + 1.64, 1.32, 0.80, 0.42, f"RS-$Q_{lab}$", COLORS["comm"], fs=7.3)
            for i, lab in enumerate(["0", "1", "2"]):
                add_box(ax, ag_start_x + i * 0.62, 0.36, 0.62, 0.42, f"AG-$Q_{lab}$", COLORS["comm"], fs=7.2)

    panel(axes[0], "(a) Standard PowerSGD: communicate both factors", half=False)
    panel(axes[1], "(b) Half-step Helix: communicate one active factor", half=True)
    fig.savefig(OUT / "fig_half_step_powersgd.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_half_step_powersgd.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_nested_prefix():
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    ax.text(2.3, 3.15, "Nested factor state", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(7.4, 3.15, "Active communication view", ha="center", va="center", fontsize=11, weight="bold")

    x0, y0 = 0.55, 2.18
    widths = [0.78, 0.78, 0.78, 0.78]
    labels = ["r=2", "r=4", "r=6", "r=8"]
    shades = ["#7D96A3", "#A9BBC1", "#C7D1D4", "#E3E0D7"]
    x = x0
    for w, lab, col in zip(widths, labels, shades):
        add_round(ax, x, y0, w, 0.48, lab, col, fs=8.6, weight="bold")
        x += w
    x = x0
    for w, lab, col in zip(widths[:2], labels[:2], shades[:2]):
        add_round(ax, x, 1.36, w, 0.48, lab, col, fs=8.6, weight="bold")
        x += w
    add_arrow(ax, (4.05, 2.32), (4.05, 1.75), lw=1.1)
    ax.text(3.45, 1.05, "rank decreases;\nstate is retained", fontsize=9, ha="center", va="center")

    add_round(ax, 5.35, 2.25, 3.9, 0.42, "Fixed allocation", COLORS["inactive"], fs=8.8, weight="bold")
    add_arrow(ax, (9.42, 2.45), (9.42, 1.72), lw=1.1)
    add_round(ax, 5.35, 1.36, 1.45, 0.48, "Active\nprefix", COLORS["active"], fs=8.6, weight="bold")
    add_round(ax, 6.80, 1.36, 0.58, 0.48, "Pad", COLORS["inactive"], fs=8.2, weight="bold")
    add_round(ax, 7.38, 1.36, 1.87, 0.48, "Inactive suffix", COLORS["suffix"], fs=8.6, weight="bold")

    ax.plot([5.38, 5.38, 7.35, 7.35], [1.12, 0.98, 0.98, 1.12], color=COLORS["edge"], linewidth=0.85)
    ax.text(6.36, 0.72, "Sent to collective", fontsize=8.8, ha="center", va="center")
    ax.text(8.32, 0.72, "Kept in memory", fontsize=8.8, ha="center", va="center")

    fig.savefig(OUT / "fig_nested_active_prefix.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_nested_active_prefix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    draw_half_step()
    draw_nested_prefix()
    print(OUT)
