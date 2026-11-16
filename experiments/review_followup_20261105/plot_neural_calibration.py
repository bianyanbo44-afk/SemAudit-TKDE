"""Same-family irregular calibration, effect response and detection probability."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'results/analysis/neural_calibration_worlds.json').is_file():
    ROOT = ROOT.parent
FIGURES = ROOT / 'figures'
FIGURES.mkdir(exist_ok=True)
sys.path.insert(0, str(Path.home() / '.codex/skills/nature-figure/scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.size': 7,
    'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
    'legend.frameon': False,
})


def wilson(count, total):
    critical = stats.norm.ppf(.975)
    fraction = count / total
    denominator = 1 + critical ** 2 / total
    center = (fraction + critical ** 2 / (2 * total)) / denominator
    half = critical * np.sqrt(fraction * (1 - fraction) / total + critical ** 2 / (4 * total ** 2)) / denominator
    return np.array([center - half, center + half])


def main():
    source = ROOT / 'results/analysis/neural_calibration_worlds.json'
    rows = json.loads(source.read_text(encoding='utf8'))
    assert len(rows) == 75
    strengths = [0., .05, .1, .25, 1.]
    families = [
        ('selective_ssm', 'Selective state-space', '#0F4D92', 'o', '-'),
        ('immtsf_recavg_gradd', 'RecAvg + GR-Add', '#D98B2B', 's', '--'),
        ('immtsf_t2v_xattn', 'T2V-XAttn + XAttn-Add', '#68717A', '^', ':'),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    figure.subplots_adjust(left=.09, right=.98, bottom=.29, top=.78, wspace=.32)
    exported = []
    for index, (family, label, color, marker, linestyle) in enumerate(families):
        means, intervals, probabilities, bounds = [], [], [], []
        for strength in strengths:
            selected = [row for row in rows if row['family'] == family and row['strength'] == strength]
            assert len(selected) == 5
            effects = np.array([row['fixed_sag'] for row in selected])
            mean = effects.mean()
            half = stats.t.ppf(.975, len(effects) - 1) * stats.sem(effects)
            means.append(mean)
            intervals.append([mean - half, mean + half])
            count = sum(row['holm_detection'] for row in selected)
            probabilities.append(100 * count / len(selected))
            bounds.append(100 * wilson(count, len(selected)))
            exported.append(dict(family=family, strength=strength, worlds=5,
                mean_fixed_sag=float(mean), world_t95_low=float(mean - half), world_t95_high=float(mean + half),
                holm_detections=count, nominal_detections=sum(row['nominal_detection'] for row in selected),
                screened_detections=sum(row['screened_detection'] for row in selected)))
        x = np.arange(len(strengths)) + (index - 1) * .075
        intervals = np.array(intervals)
        bounds = np.array(bounds)
        means = np.array(means)
        probabilities = np.array(probabilities)
        axes[0].errorbar(x, means, yerr=np.array([means - intervals[:, 0], intervals[:, 1] - means]),
                         color=color, marker=marker, linestyle=linestyle, ms=3.5, lw=1.1, capsize=1.8, label=label)
        axes[1].errorbar(x, probabilities,
                         yerr=np.maximum(0, np.array([probabilities - bounds[:, 0], bounds[:, 1] - probabilities])),
                         color=color, marker=marker, linestyle=linestyle, ms=3.5, lw=1.1, capsize=1.8)
    for index, (axis, title) in enumerate(zip(axes, ['Same-checkpoint alignment response', 'Multiplicity-adjusted detection'])):
        axis.set_title(title, loc='left', fontsize=8, fontweight='bold', pad=15)
        axis.text(-.16, 1.14, chr(97 + index), transform=axis.transAxes, fontsize=10, fontweight='bold')
        axis.set_xticks(np.arange(len(strengths)), ['0', '.05', '.1', '.25', '1'])
        axis.set_xlabel('Injected signal strength')
        axis.axhline(0, color='#B8BEC6', lw=.6, zorder=0)
        axis.tick_params(length=3)
    axes[0].set_ylabel('Aligned-over-history-rotation gain (%)')
    axes[1].set_ylabel('Detected generated worlds (%)')
    axes[1].set_ylim(-4, 104)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .015), ncol=3, fontsize=6.5,
                  handlelength=2., columnspacing=1.7)
    require_matplotlib_panel_alignment(figure, axes=axes, panel_ids=['a', 'b'], row_groups=[['a', 'b']],
        json_out=FIGURES / 'fig_neural_calibration_alignment.json', strict=True)
    figure.savefig(FIGURES / 'fig_neural_calibration.pdf', bbox_inches='tight', pad_inches=.05)
    figure.savefig(FIGURES / 'fig_neural_calibration.svg', bbox_inches='tight', pad_inches=.05)
    figure.savefig(FIGURES / 'fig_neural_calibration.png', bbox_inches='tight', pad_inches=.05, dpi=600)
    plt.close(figure)
    (FIGURES / 'fig_neural_calibration_data.json').write_text(json.dumps(exported, indent=2), encoding='utf8')
    contract = dict(question='How do the original three neural families respond to known irregular text signal strength?',
        panels=dict(a='Magnitude and cross-world variability of paired fixed-checkpoint response',
                    b='Probability of directional detection under the declared multiplicity adjustment'),
        all_declared_cells_retained=True, worlds_per_cell=5,
        error_bars=dict(a='Student t 95% interval across five independently generated worlds',
                        b='Wilson 95% interval across five generated worlds'),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (FIGURES / 'fig_neural_calibration_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
