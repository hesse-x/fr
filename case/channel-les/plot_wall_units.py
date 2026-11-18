"""Plot instantaneous plane-mean y+–U+ profiles from a channel checkpoint."""
import argparse
import csv
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import torch
from fr import CartesianMesh
from .solve import ChannelLES

wall_scales = import_module('case.channel-dns.postprocess').wall_scales


@torch.no_grad()
def plot_checkpoint(checkpoint, output=None):
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    mesh = CartesianMesh(config['mesh_edges']) if 'mesh_edges' in config else None
    model = ChannelLES(tuple(config['cells']), p=config['p'], mesh=mesh,
        re_tau=config['re_tau'], mach=config['mach'], lengths=config['lengths'],
        bulk_velocity=config['bulk_velocity'], forcing=0.)
    profile = model.plane_mean(model.field_moments(saved['U']))
    shear, density, friction = wall_scales(model, profile)
    reynolds = friction*density/model.mu
    y = model.s.coords[1][0, :, 0, 0, :, 0]
    out = Path(output) if output is not None else checkpoint.parent/f'snapshot-{saved["step"]:06d}'
    out.mkdir(parents=True, exist_ok=True)
    profiles, rows = [], []
    for side, label in enumerate(('lower', 'upper')):
        mask = y < 0 if side == 0 else y > 0
        distance = 1+y[mask] if side == 0 else 1-y[mask]
        yp = distance*friction[side]*density[side]/model.mu
        up = profile[..., 1][mask]/friction[side]
        order = torch.argsort(yp)
        yp, up, distance = yp[order].numpy(), up[order].numpy(), distance[order].numpy()
        profiles.append((label, yp, up))
        rows.extend((label, float(d), float(a), float(b)) for d, a, b in zip(distance, yp, up))
    with (out/'wall_units.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('wall', 'wall_distance', 'y_plus', 'u_plus'))
        writer.writerows(rows)
    metrics = dict(step=saved['step'], time=saved['time'], checkpoint=str(checkpoint.resolve()),
        averaging='instantaneous x-z plane mean; no time averaging',
        wall_order=['lower', 'upper'], tau_wall=shear.tolist(), rho_wall=density.tolist(),
        u_tau=friction.tolist(), re_tau=reynolds.tolist(),
        wall_shear_method='molecular viscosity times endpoint derivative of plane-mean velocity polynomial',
        cells=config['cells'], p=config['p'], nominal_mach=config['mach'])
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    for side, (label, yp, up) in enumerate(profiles):
        ax.semilogx(yp, up, 'o-', markersize=3.5, linewidth=1.4,
                    label=f'{label.capitalize()} wall, '+r'$Re_\tau$'+f'={float(reynolds[side]):.2f}')
    reference_path = Path(__file__).resolve().parents[1]/'channel-dns'/'reference'/'chan180.means'
    reference = np.loadtxt(reference_path, comments='#')
    positive = reference[:, 1] > 0
    ax.semilogx(reference[positive, 1], reference[positive, 2], 'k--', linewidth=1.5,
                label=r'MKM1999 time mean, $Re_\tau=178.12$')
    viscous = np.geomspace(.5, 5., 100)
    ax.semilogx(viscous, viscous, ':', color='.5', label=r'$U^+=y^+$')
    ax.set(xlabel=r'$y^+$', ylabel=r'$U^+$', xlim=(.5, max(float(reynolds.max()), 180.)*1.05),
           ylim=(0, max(max(float(up.max()) for _, _, up in profiles), 18.)+1),
           title=f'Instantaneous plane-mean velocity | step {saved["step"]:,}, t={saved["time"]:.4f}')
    ax.grid(True, which='both', alpha=.25)
    ax.legend(fontsize=9)
    fig.savefig(out/'y_plus_u_plus.png', dpi=220)
    fig.savefig(out/'y_plus_u_plus.pdf')
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f'Output: {out}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('positive threads required')
    torch.set_num_threads(args.threads)
    plot_checkpoint(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
