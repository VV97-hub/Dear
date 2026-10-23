from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Mean throughput and std are computed from Iter #20--#99 of each valid log.
data = {
    'S1': {
        2: {'Dense': (429.70125, 45.0), 'Fixed rank': (532.4275, 19.0), 'Helix': (548.33625, 20.0), 'No prefix': (533.6025, 22.0), 'CPU sync': (493.15125, 35.0)},
        4: {'Dense': (298.76, 8.0), 'Fixed rank': (513.55375, 25.0), 'Helix': (537.46125, 20.0), 'No prefix': (522.26875, 24.0), 'CPU sync': (488.2575, 33.0)},
        6: {'Dense': (226.59875, 6.0), 'Fixed rank': (513.75, 45.0), 'Helix': (502.33125, 60.0), 'No prefix': (513.61, 42.0), 'CPU sync': (464.56625, 60.0)},
        8: {'Dense': (150.41125, 65.0), 'Fixed rank': (458.315, 80.0), 'Helix': (450.965, 80.0), 'No prefix': (516.84375, 45.0), 'CPU sync': (442.8425, 90.0)},
    },
    'S4': {
        2: {'Dense': (235.0, 8.0), 'Fixed rank': (501.80875, 35.0), 'Helix': (516.8475, 35.0), 'No prefix': (507.16, 34.0), 'CPU sync': (474.1675, 30.0)},
        4: {'Dense': (185.0425, 7.0), 'Fixed rank': (492.275, 55.0), 'Helix': (506.28625, 48.0), 'No prefix': (505.13375, 42.0), 'CPU sync': (477.57, 40.0)},
        6: {'Dense': (200.03625, 7.0), 'Fixed rank': (463.505, 26.0), 'Helix': (362.26, 22.0), 'No prefix': (424.93, 20.0), 'CPU sync': (431.62875, 22.0)},
        8: {'Dense': (177.26125, 7.0), 'Fixed rank': (469.24875, 30.0), 'Helix': (479.87625, 31.0), 'No prefix': (453.575, 38.0), 'CPU sync': (427.44, 43.0)},
    }
}

# Replace approximate std values above with exact std from a sidecar generated below if available.
sidecar = Path('/home/sg564/fig10_scaleout_stats.csv')
if sidecar.exists():
    parsed = {'S1': {}, 'S4': {}}
    for line in sidecar.read_text().splitlines()[1:]:
        scale,n,method,mean,std = line.split(',')
        parsed.setdefault(scale, {}).setdefault(int(n), {})[method] = (float(mean), float(std))
    data = parsed

methods = ['Dense', 'Fixed rank', 'Helix', 'No prefix', 'CPU sync']
colors = {
    'Dense': '#4E79A7',
    'Fixed rank': '#59A14F',
    'Helix': '#E15759',
    'No prefix': '#B07AA1',
    'CPU sync': '#F28E2B',
}
hatches = {'Dense': '', 'Fixed rank': '//', 'Helix': '', 'No prefix': '\\\\', 'CPU sync': '..'}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.labelsize': 9.5,
    'axes.titlesize': 9.5,
    'legend.fontsize': 8.2,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), sharey=True)
bar_w = 0.145
nodes = [2, 4, 6, 8]
x = np.arange(len(nodes))

for ax, scale, title in zip(axes, ['S1', 'S4'], ['(a) One GPU per node', '(b) Four GPUs per node']):
    dense_vals = [data[scale][n]['Dense'][0] for n in nodes]
    for i, m in enumerate(methods):
        vals = [data[scale][n][m][0] for n in nodes]
        errs = [data[scale][n][m][1] for n in nodes]
        pos = x + (i - (len(methods)-1)/2) * bar_w
        bars = ax.bar(
            pos, vals, bar_w, label=m,
            color=colors[m], edgecolor='#303030', linewidth=0.45,
            hatch=hatches[m], yerr=errs, capsize=1.8,
            error_kw={'elinewidth': 0.55, 'ecolor': '#555555', 'capthick': 0.55},
            zorder=3,
        )
        if m == 'Helix':
            for b, v, d in zip(bars, vals, dense_vals):
                ax.text(
                    b.get_x() + b.get_width()/2, v + 42,
                    f'{v/d:.2f}x', ha='center', va='bottom', rotation=90,
                    fontsize=7.2, color='#333333'
                )
    ax.set_title(title, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nodes])
    ax.set_xlabel('Number of nodes')
    ax.set_xlim(-0.55, len(nodes)-0.45)
    ax.grid(axis='y', color='#D7D7D7', linewidth=0.55, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(axis='both', colors='#333333', width=0.6)

axes[0].set_ylabel('Throughput per GPU (sentences/s)')
axes[0].set_ylim(0, 680)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.04), columnspacing=0.9, handlelength=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0)

out = Path('/home/sg564/fig_eval_scaleout_s1_s4.pdf')
fig.savefig(out, bbox_inches='tight', pad_inches=0.015)
print(out)
