"""Compare completed phase cases: python -m case.uniform_array.compare."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from .solve import PHASE_PRESETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='results/uniform_array/phases')
    args = parser.parse_args()
    root = Path(args.root)
    # Read all inputs before writing: missing/failed cases must not produce a
    # deceptively complete comparison.
    records = []
    for name, phases in PHASE_PRESETS.items():
        path = root/name
        stats = json.loads((path/'metrics.json').read_text())
        if stats.get('preset') != name or not stats.get('validation_passed'):
            raise ValueError(f'{name}: expected a validated matching preset run')
        with (path/'pattern.csv').open() as f:
            rows = list(csv.DictReader(f))
        records.append((name, phases, stats, rows))

    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'fr-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    summary = []
    for ax, (name, phases, stats, rows) in zip(axes.flat, records):
        angles = np.array([float(r['theta_degrees']) for r in rows])
        power = np.array([float(r['fr_normalized_power']) for r in rows])
        reference = np.array([float(r['analytic_normalized_power']) for r in rows])
        ax.plot(angles, 10*np.log10(np.maximum(power, 1e-5)), label='FR')
        ax.plot(angles, 10*np.log10(np.maximum(reference, 1e-5)), '--', label='Analytic')
        ax.set(xlim=(-180, 180), ylim=(-40, 1), xlabel='Angle from +y toward +x (degrees)',
               ylabel='Normalized power (dB)', title=f'{name}: {list(phases)} deg')
        ax.grid(alpha=.3)
        ax.legend()
        summary.append(dict(preset=name, phases_degrees=phases,
                            pattern_relative_l2=stats['pattern_relative_l2'],
                            front_peak_angle_degrees=stats['front_peak_angle_degrees'],
                            broadside_normalized_power=stats['broadside_normalized_power'],
                            cycle_change=stats['surface_cycle_relative_change']))
    fig.savefig(root/'comparison.png', dpi=160)
    plt.close(fig)
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
