"""Compare exported wall-unit profiles with the original MKM1999 mean table."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='run directory containing postprocess/')
    args = parser.parse_args()
    out = args.run/'postprocess'
    mach = json.loads((args.run/'run.json').read_text())['config']['mach']
    metrics = json.loads((out/'metrics.json').read_text())
    with (out/'wall_units.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    reference_path = Path(__file__).with_name('reference')/'chan180.means'
    reference = np.loadtxt(reference_path, comments='#')
    ref_y, ref_u = reference[:, 1], reference[:, 2]
    profiles, comparisons, samples = {}, [], []
    for side, wall in enumerate(('lower', 'upper')):
        points = sorted((float(r['y_plus']), float(r['u_plus'])) for r in rows if r['wall'] == wall)
        yp, up = np.array(points).T
        profiles[wall] = (yp, up)
        for x, u in zip(yp, up):
            if ref_y[0] <= x <= ref_y[-1]:
                target = float(np.interp(x, ref_y, ref_u))
                comparisons.append((wall, x, u, target, u-target))
        for target_y in (1., 5., 10., 30., 50., 100., 150.):
            if yp[0] <= target_y <= min(yp[-1], ref_y[-1]):
                u = float(np.interp(target_y, yp, up))
                target = float(np.interp(target_y, ref_y, ref_u))
                samples.append(dict(wall=wall, y_plus=target_y, u_plus=u,
                                    reference_u_plus=target, difference=u-target))
    with (out/'mkm_comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('wall', 'y_plus', 'u_plus', 'mkm_u_plus', 'difference'))
        writer.writerows(comparisons)
    summary = dict(sample_start_step=metrics['sample_start_step'], end_step=metrics['step'],
                   nominal_mach=mach,
                   simulation_re_tau=metrics['re_tau'], reference_re_tau=178.12,
                   reference_file=str(reference_path.resolve()),
                   comparison='same y+, separate actual wall scales; different Re_tau and Mach number',
                   interpolation='linear in y+ for both simulation and reference', samples=samples)
    (out/'mkm_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, residual) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.6, 1]})
    colors = ('#1764ab', '#e17c19')
    for side, (wall, (yp, up)) in enumerate(profiles.items()):
        dense = np.geomspace(yp[0], yp[-1], 600)
        label = f'{wall.capitalize()} wall, '+r'$Re_\tau$'+f'={metrics["re_tau"][side]:.2f}'
        ax.semilogx(dense, np.interp(dense, yp, up), color=colors[side], label=label)
        ax.semilogx(yp, up, 'o', color=colors[side], markersize=2.7)
        data = np.array([r[1:] for r in comparisons if r[0] == wall])
        residual.semilogx(data[:, 0], data[:, 3], '.-', color=colors[side], label=wall.capitalize())
    ax.semilogx(ref_y[1:], ref_u[1:], 'k--', linewidth=1.6,
                label=r'MKM1999, $Re_\tau=178.12$')
    ax.set(xlabel=r'$y^+$', ylabel=r'$U^+$', xlim=(.4, 200), ylim=(0, 20),
           title='Plane/time mean velocity')
    ax.legend(fontsize=9)
    residual.axhline(0, color='black', linestyle='--', linewidth=1)
    residual.set(xlabel=r'$y^+$', ylabel=r'$U^+_{simulation}-U^+_{MKM}$', xlim=(.4, 200),
                 title='Difference at the same '+r'$y^+$')
    residual.legend(fontsize=9)
    for axis in (ax, residual):
        axis.grid(True, which='both', alpha=.25)
    fig.suptitle(f'Mean over steps {metrics["sample_start_step"]:,}–{metrics["step"]:,}; nominal Ma={mach:g}', fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out/f'u_plus_mkm_comparison.{extension}', dpi=200)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
