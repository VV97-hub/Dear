from pathlib import Path
import csv, math
import matplotlib.pyplot as plt

src = Path('/home/sg564/gpt_loss_20260713')
files = {
    'Dense': src / 'GPTC0_gpt230m_dense_long_%j.csv',
    'Fixed rank': src / 'GPTC1_gpt230m_fixed_long_%j.csv',
    'Helix': src / 'GPTC9_gpt230m_helix_long_%j.csv',
    'Full PowerSGD': src / 'GPTC8_gpt230m_fullpq_long_%j.csv',
}
colors = {'Dense':'#4E79A7','Fixed rank':'#59A14F','Helix':'#E15759','Full PowerSGD':'#8A6F3D'}
linestyles = {'Dense':'-','Fixed rank':'--','Helix':'-','Full PowerSGD':'-.'}

def read_csv(path):
    rows=[]
    with path.open(newline='') as f:
        for r in csv.DictReader(f):
            try:
                step=int(float(r['step'])); loss=float(r['loss'])
            except Exception:
                continue
            if math.isfinite(loss): rows.append((step,loss))
    return rows

series={k:read_csv(v) for k,v in files.items()}
min_common=max(min(s for s,_ in rows) for rows in series.values())
max_common=min(max(s for s,_ in rows) for rows in series.values())
for k in list(series):
    series[k]=[(s,y) for s,y in series[k] if min_common <= s <= max_common]

def moving_average(vals, win=7):
    out=[]
    for i in range(len(vals)):
        lo=max(0,i-win//2); hi=min(len(vals),i+win//2+1)
        out.append(sum(vals[lo:hi])/(hi-lo))
    return out

plt.rcParams.update({
    'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
    'font.size':8.8,'axes.labelsize':9.2,'legend.fontsize':8.2,
    'xtick.labelsize':8.4,'ytick.labelsize':8.4,'pdf.fonttype':42,'ps.fonttype':42,
})
fig, ax = plt.subplots(figsize=(3.95,2.35))
for name, rows in series.items():
    xs=[(s-min_common)/1000.0 for s,_ in rows]
    ys=[y for _,y in rows]
    smooth=moving_average(ys, win=7)
    ax.plot(xs, ys, color=colors[name], alpha=0.14, linewidth=0.55)
    ax.plot(xs, smooth, label=name, color=colors[name], linestyle=linestyles[name], linewidth=1.35)
ax.set_xlabel('Post-warmup training steps (×10$^3$)')
ax.set_ylabel('Training loss')
ax.set_xlim(0,(max_common-min_common)/1000.0)
ax.set_ylim(0,6.1)
ax.grid(axis='y',color='#D8D8D8',linewidth=0.55)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#666666'); ax.spines['bottom'].set_color('#666666')
ax.legend(loc='upper right', frameon=False, ncol=1, handlelength=2.0)
fig.tight_layout(pad=0.35)
out=Path('/home/sg564/fig_eval_gpt_loss_full.pdf')
fig.savefig(out,bbox_inches='tight',pad_inches=0.015)
print('wrote', out, 'common_step_range', min_common, max_common, 'formal_k', (max_common-min_common)/1000.0)
