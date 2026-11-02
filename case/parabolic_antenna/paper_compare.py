"""Compare native E/H cuts with digitized Ban et al. (2026), Figure 16.

This is a comparison of related source/PEC models, not an exact reproduction
of the paper's unspecified FEM voltage edge or reflector backside.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from case.uniform_array.solve import far_field

PAPER_URL = 'https://doi.org/10.3390/math14101706'
REFERENCE = Path(__file__).with_name('reference')/'ban2026_figure16.csv'


def beamwidth(theta, power):
    """Connected -3 dB lobe about the front peak, with linear crossings."""
    front = np.flatnonzero(np.abs(theta) <= 90)
    peak = int(front[np.argmax(power[front])])
    half = power[peak]/2
    left = right = peak
    while left > 0 and power[left] >= half:
        left -= 1
    while right < len(power)-1 and power[right] >= half:
        right += 1
    if left == 0 or right == len(power)-1:
        return None
    a = theta[left]+(half-power[left])/(power[left+1]-power[left])*(theta[left+1]-theta[left])
    b = theta[right-1]+(half-power[right-1])/(power[right]-power[right-1])*(theta[right]-theta[right-1])
    return float(b-a)


def compare(solution, *, sphere=False):
    torch.set_num_threads(4)
    data = torch.load(solution, map_location='cpu', weights_only=True)
    stats = data['stats']
    if not (math.isclose(stats['diameter'], 2.) and math.isclose(stats['focal_length'], .5)):
        raise ValueError('paper Case 4 requires D/lambda=2 and f/lambda=0.5')
    out = Path(solution).parent
    reference = np.genfromtxt(REFERENCE, delimiter=',', names=True)
    theta = torch.linspace(-180., 180., 721, dtype=torch.float64)
    angle = theta*math.pi/180
    cuts = {}
    for plane, directions in (
        ('E', torch.stack((torch.zeros_like(angle), angle.cos(), angle.sin()), -1)),
        ('H', torch.stack((angle.sin(), angle.cos(), torch.zeros_like(angle)), -1))):
        field = far_field(data['surface_xyz'], data['surface_normals'], data['surface_weights'],
                          data['surface_fields'], directions)
        power = field.abs().square().sum(-1).numpy()
        cuts[plane] = power/power.max()
    angles = theta.numpy()
    voltage_feed = stats.get('feed') == 'voltage'
    differences = [
        (f"Paper: prescribed FEM edge voltage via penalty; native: mass-weighted DG line-voltage constraint, edge length {stats.get('voltage_length')} wavelength (paper edge length unspecified)."
         if voltage_feed else 'Paper: Y-polarized FEM voltage edge at (0,0,0.5); native: normalized compact spherical electric current.'),
        'Paper axis +z and feed polarization +y are mapped to native axis +y and polarization +z.',
        'Paper: PEC upper surface, backside unspecified; native: finite-thickness stair-step PEC shell.']
    if stats.get('outer_boundary') == 'silver-muller':
        differences.append('Both use a 3-wavelength cube and first-order Silver-Muller condition; FEM boundary integral and FR outgoing-characteristic implementations differ.')
    else:
        differences.append('Paper: 3-wavelength box with Silver-Muller ABC; native: free-space box with split PML.')
    summary = dict(paper=PAPER_URL, figure=16, source='publisher PDF raster, digitized curves; not original numerical data',
        geometry=dict(diameter_wavelengths=2., focal_length_wavelengths=.5),
        model_differences=differences, native=stats, planes={})
    for plane, power in cuts.items():
        values = dict(native_hpbw_degrees=beamwidth(angles, power),
                      native_peak_degrees=float(angles[np.argmax(power)]))
        for method in ('vfem', 'meep'):
            db = reference[f'{plane}_{method}_db']
            valid = np.isfinite(db)
            t, ref = reference['theta_degrees'][valid], 10**(db[valid]/10)
            native = np.interp(t, angles, power)
            main = np.abs(t) <= 50
            front = np.abs(t) <= 90
            values[method] = dict(digitized_hpbw_degrees=beamwidth(t, ref),
                front_power_relative_l2=float(np.linalg.norm(native[front]-ref[front])/np.linalg.norm(ref[front])),
                main_lobe_db_rmse=float(np.sqrt(np.mean((10*np.log10(np.maximum(native[main], 1e-12))-db[valid][main])**2))))
        summary['planes'][plane] = values
    if sphere:
        # Independent full-sphere solid-angle integration, y is the polar axis.
        mu, weights = np.polynomial.legendre.leggauss(32)
        phi = np.arange(64)*2*np.pi/64
        u, az = np.meshgrid(mu, phi, indexing='ij')
        radial = np.sqrt(1-u*u)
        directions = torch.tensor(np.stack((radial*np.cos(az), u, radial*np.sin(az)), -1).reshape(-1, 3))
        field = far_field(data['surface_xyz'], data['surface_normals'], data['surface_weights'],
                          data['surface_fields'], directions)
        power = field.abs().square().sum(-1).numpy().reshape(32, 64)
        integral = float(np.sum(weights[:, None]*power)*2*np.pi/64)
        bore = far_field(data['surface_xyz'], data['surface_normals'], data['surface_weights'],
                         data['surface_fields'], torch.tensor([[0., 1., 0.]], dtype=torch.float64))
        boresight = float(bore.abs().square().sum())
        summary['boresight_directivity_dbi'] = 10*math.log10(4*math.pi*boresight/integral)
        summary['paper_reported_peak_db'] = dict(vfem=11.9860, meep=12.4062)
        summary['absolute_metric_note'] = 'Native is radiated-power-normalized directivity, not port gain; paper port normalization is not fully specified.'
    with (out/'paper_patterns.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(('theta_degrees', 'native_E_power', 'native_H_power'))
        writer.writerows(zip(angles, cuts['E'], cuts['H']))
    (out/'paper_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, (plane, power) in zip(axes, cuts.items()):
        ax.plot(angles, 10*np.log10(np.maximum(power, 1e-6)), color='black', label='Native FR ('+('voltage edge, local grid)' if voltage_feed else 'current source, stair-step PEC)'))
        for method, color, style in (('vfem', 'tab:blue', '--'), ('meep', 'tab:red', ':')):
            ax.plot(reference['theta_degrees'], reference[f'{plane}_{method}_db'], style,
                    color=color, lw=1.5, label=f'Paper {method.upper()} (digitized)')
        ax.set(xlim=(-180, 180), ylim=(-40, 2), xlabel='Angle from +y (degrees)',
               ylabel='Normalized power (dB)', title=f'{plane}-plane: '+('yz cut' if plane=='E' else 'xy cut'))
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc='lower center')
    fig.suptitle('Ban et al. (2026), Case 4: D=2 wavelengths, f=0.5 wavelength\n'+(('Voltage-edge constraint; '+('Silver-Muller box' if stats.get('outer_boundary') == 'silver-muller' else 'PML exterior')+'; stair-step PEC') if voltage_feed else 'Feed and PEC geometry approximations differ; curves are not an exact-model error test'))
    fig.savefig(out/'paper_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    parser.add_argument('--sphere', action='store_true', help='also integrate 32x64 full-sphere directions for directivity')
    args = parser.parse_args()
    compare(args.solution, sphere=args.sphere)


if __name__ == '__main__':
    main()
