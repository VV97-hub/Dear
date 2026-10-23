"""Regenerate the handoff figure, changing only two labels before saving."""
from pathlib import Path
import importlib.util
import sys
sys.dont_write_bytecode = True

root = Path(__file__).parent
original = Path('/mnt/c/Users/sg564/Desktop/Helix技术论文/实验文件/Figures_20260624/generate_acpr_experiment_figures_20260624.py')
spec = importlib.util.spec_from_file_location('original_plot', original)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.ROOT = Path('/mnt/c/Users/sg564/Desktop/Helix技术论文/实验文件')
mod.OUT = root
mod.configure_plot_style()

def save_with_corrected_labels(fig, stem):
    assert stem == 'fig5_stream_handoff_distribution'
    fig.savefig(root / 'handoff_reproduced_before.pdf', bbox_inches='tight', pad_inches=0.02)
    ax = fig.axes[0]
    assert ax.get_title() == '(a) Host-side RS launch delay'
    assert ax.get_xlabel() == 'Buffer ready $\\rightarrow$ RS enqueued (ms)'
    ax.set_title('(a) Host-side RS issue delay')
    ax.set_xlabel('Host group-ready $\\rightarrow$ RS issue (ms)')
    fig.savefig(root / 'source/figures/fig_eval_handoff.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(root / 'handoff_corrected.png', dpi=200, bbox_inches='tight', pad_inches=0.02)
    mod.plt.close(fig)

mod.save = save_with_corrected_labels
mod.plot_handoff_distribution()
print('Replotted from the original samples; only left title and x-axis label changed.')
