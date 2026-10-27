"""Export time-mean fields and wall-scaled profiles from long_run checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from channel import Channel


def wall_scales(model, profile):
    """Wall shear from endpoint derivatives of the mean velocity polynomial."""
    derivative = (profile[..., 1] @ model.s.D.T)*2/model.s.widths[1][:, None]
    shear = model.mu*torch.stack((derivative[0] @ model.s.L,
                                  -(derivative[-1] @ model.s.R)))
    density = torch.stack((profile[0, :, 0] @ model.s.L,
                           profile[-1, :, 0] @ model.s.R))
    if not bool((shear > 0).all() and (density > 0).all()):
        raise ValueError('nonpositive wall shear/density; cannot define friction scales')
    return shear, density, (shear/density).sqrt()


def write_csv(path, header, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    duration = saved['sample_duration']
    if duration <= 0:
        parser.error('checkpoint has no accumulated mean field')
    config = saved['config']
    model = Channel(tuple(config['cells']), p=config['p'], stretch=config['stretch'],
                    re_tau=config['re_tau'], mach=config['mach'],
                    lengths=config.get('lengths', (2*math.pi, math.pi)),
                    bulk_velocity=config.get('bulk_velocity', 15.))
    out = args.output or args.checkpoint.parent/'postprocess'
    out.mkdir(parents=True, exist_ok=True)
    field = saved['field_integral']/duration
    profile = model.plane_mean(field)
    shear, density, friction = wall_scales(model, profile)
    y = model.s.coords[1][0, :, 0, 0, :, 0]
    velocity = profile[..., 1:4]
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    stress = profile[..., 6:].clone()
    for k, (i, j) in enumerate(pairs):
        stress[..., k] -= velocity[..., i]*velocity[..., j]
    rms = stress[..., :3].clamp_min(0).sqrt()
    full = torch.cat((y[..., None], profile[..., :6], rms, stress), -1)
    write_csv(out/'profiles.csv',
              ['y', 'rho', 'u', 'v', 'w', 'pressure', 'temperature',
               'u_rms', 'v_rms', 'w_rms', 'uu', 'vv', 'ww', 'uv', 'uw', 'vw'],
              full.reshape(-1, full.shape[-1]).tolist())
    rows = []
    for side, label in enumerate(('lower', 'upper')):
        mask = y < 0 if side == 0 else y > 0
        distance = 1+y[mask] if side == 0 else 1-y[mask]
        yp = distance*friction[side]*density[side]/model.mu
        up = velocity[..., 0][mask]/friction[side]
        order = torch.argsort(yp)
        rows.extend((label, float(distance[i]), float(yp[i]), float(up[i])) for i in order)
    write_csv(out/'wall_units.csv', ['wall', 'wall_distance', 'y_plus', 'u_plus'], rows)
    reynolds = field[..., 6:].clone()
    for k, (i, j) in enumerate(pairs):
        reynolds[..., k] -= field[..., 1+i]*field[..., 1+j]
    torch.save(dict(density=field[..., 0], velocity=field[..., 1:4],
                    pressure=field[..., 4], temperature=field[..., 5],
                    reynolds_stress=reynolds,
                    reynolds_components=['uu', 'vv', 'ww', 'uv', 'uw', 'vw'],
                    edges=model.s.edges, config=config,
                    sample_start=saved['time']-duration, sample_end=saved['time'],
                    duration=duration, averaging='Reynolds time average; trapezoidal integration'),
               out/'mean_fields.pt')
    metrics = dict(step=saved['step'], sample_start_step=config['sample_start_step'],
                   sample_duration=duration, sample_start=saved['time']-duration,
                   sample_end=saved['time'], mu=model.mu,
                   tau_wall=shear.tolist(), rho_wall=density.tolist(),
                   u_tau=friction.tolist(),
                   re_tau=(friction*density/model.mu).tolist(),
                   wall_order=['lower', 'upper'],
                   wall_shear_method='mu times endpoint derivative of plane/time mean velocity polynomial',
                   averaging='Reynolds plane/time mean; separate walls, no symmetry assumption',
                   statistical_convergence_verified=False)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label in ('lower', 'upper'):
        points = [r for r in rows if r[0] == label]
        ax.semilogx([r[2] for r in points], [r[3] for r in points], '.-',
                    label=f'{label} wall, Re_tau={float(metrics["re_tau"][0 if label == "lower" else 1]):.1f}')
    limit = max(r[2] for r in rows)
    reference_path = Path(__file__).with_name('reference')/'chan180.means'
    if reference_path.exists():
        reference = [[float(v) for v in line.split()] for line in reference_path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
        reference = [r for r in reference if r[1] > 0]
        ax.semilogx([r[1] for r in reference], [r[2] for r in reference],
                    'k--', label='Moser et al. (1999), Re_tau=178.12')
    viscous = torch.logspace(-1, min(torch.log10(torch.tensor(5.)).item(),
                                   torch.log10(torch.tensor(limit)).item()), 100)
    ax.semilogx(viscous, viscous, '--', label=r'$U^+=y^+$')
    ax.set(xlabel=r'$y^+$', ylabel=r'$U^+$', title='Plane/time mean velocity (different Re_tau)')
    ax.grid(True, which='both', alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/'u_plus.png', dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f'Output: {out}')


if __name__ == '__main__':
    main()
