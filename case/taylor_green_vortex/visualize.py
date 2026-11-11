"""Render one saved TGV snapshot: spectra and Q-criterion isosurface.

Uses the existing binary legacy VTK and spectrum CSV; never reruns the solver.
scikit-image is needed only for marching cubes, not simulation or spectra.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


def read_snapshot(path):
    """Memory-map this case's binary structured-points VTK, preserving xyz axes."""
    path = Path(path)
    fields = {}
    with path.open('rb') as stream:
        if not stream.readline().startswith(b'# vtk DataFile Version'):
            raise ValueError('expected legacy VTK')
        stream.readline()
        if stream.readline().strip() != b'BINARY':
            raise ValueError('expected binary VTK')
        if stream.readline().strip() != b'DATASET STRUCTURED_POINTS':
            raise ValueError('expected structured points')
        dimensions = tuple(map(int, stream.readline().split()[1:]))
        origin = tuple(map(float, stream.readline().split()[1:]))
        spacing = tuple(map(float, stream.readline().split()[1:]))
        if int(stream.readline().split()[1]) != math.prod(dimensions):
            raise ValueError('point count does not match dimensions')
        nx, ny, nz = dimensions
        while True:
            line = stream.readline()
            if not line:
                break
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == b'SCALARS' and tokens[2:] == [b'double', b'1']:
                components = 1
                if stream.readline().strip() != b'LOOKUP_TABLE default':
                    raise ValueError('expected scalar lookup table')
            elif tokens[0] == b'VECTORS' and tokens[2:] == [b'double']:
                components = 3
            else:
                raise ValueError(f'unsupported VTK field: {line!r}')
            offset = stream.tell()
            array = np.memmap(path, dtype='>f8', mode='r', offset=offset,
                              shape=(nz, ny, nx, components))
            fields[tokens[1].decode()] = array.transpose(2, 1, 0, 3)
            stream.seek(offset+nx*ny*nz*components*8)
    return fields, dict(dimensions=dimensions, origin=origin, spacing=spacing)


def plot_spectra(prefix, output):
    import matplotlib.pyplot as plt
    prefix, output = Path(prefix), Path(output)
    with prefix.with_name(prefix.name+'.spectrum.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    metadata = json.loads(prefix.with_name(prefix.name+'.sampling.json').read_text())
    k = np.array([int(r['wavenumber']) for r in rows])
    velocity = np.array([float(r['velocity_energy']) for r in rows])
    weighted = np.array([float(r['density_weighted_energy']) for r in rows])
    complete = np.array([r['inside_nyquist_sphere'].lower() == 'true' for r in rows])
    keep = complete & (k >= 2)
    x, ev, ew = k[keep], velocity[keep], weighted[keep]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))
    fig.suptitle(f'Taylor–Green vortex  |  t = {metadata["time"]:g}  |  Re = 1600, Ma = 0.3', fontsize=13)
    axes[0].loglog(x, ev, color='#1667a7', linewidth=1.9, label=r'Velocity $E_u(k)$')
    axes[0].loglog(x, ew, '--', color='#e1812c', linewidth=1.4, label=r'Density-weighted $E_{\sqrt{\rho}u}(k)$')
    anchor = 12
    anchor_e = velocity[np.flatnonzero(k == anchor)[0]]
    guide_x = np.array([6., 30.])
    axes[0].loglog(guide_x, anchor_e*(guide_x/anchor)**(-5/3), ':', color='.4', linewidth=1.5,
                   label=r'$k^{-5/3}$ slope guide')
    axes[0].set(xlabel=r'Wavenumber $kL$', ylabel=r'Shell kinetic energy $E(k)/U_0^2$',
                title='Energy spectrum', xlim=(1.7, 128))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].semilogx(x, x**(5/3)*ev, color='#1667a7', linewidth=1.8)
    axes[1].semilogx(x, x**(5/3)*ew, '--', color='#e1812c', linewidth=1.3)
    axes[1].set(xlabel=r'Wavenumber $kL$', ylabel=r'$k^{5/3} E(k)$',
                title='Compensated spectrum', xlim=(1.7, 128), ylim=(0, None))
    for ax in axes:
        ax.grid(which='major', color='.88', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .018, '256³ uniform samples  •  complete shells k = 2…127  •  no smoothing  •  slope guide is not reference DNS data',
             ha='center', fontsize=8, color='.35')
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(output/'energy_spectrum_t10.png', dpi=220)
    fig.savefig(output/'energy_spectrum_t10.pdf')
    plt.close(fig)
    return dict(time=metadata['time'], uniform_shape=metadata['uniform_shape'],
                velocity_spectral_sum=float(velocity.sum()), density_weighted_spectral_sum=float(weighted.sum()),
                sampled_kinetic_energy=metadata['sampled_kinetic_energy'],
                parseval_relative_error=float(abs(weighted.sum()/metadata['sampled_kinetic_energy']-1)),
                displayed_shells=[int(x[0]), int(x[-1])],
                reference_spectrum='No matching published numeric t=10 reference spectrum included',
                slope_guide='k^(-5/3), anchored at k=12; not a DNS reference',
                energy_fraction_above_k64=float(velocity[k > 64].sum()/velocity.sum()),
                energy_fraction_above_k128=float(velocity[k > 128].sum()/velocity.sum()))


def render_vortices(prefix, output, *, level=2.5, step=2):
    import matplotlib.pyplot as plt
    from matplotlib import colors, colormaps
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.ndimage import map_coordinates
    from skimage.measure import marching_cubes
    prefix, output = Path(prefix), Path(output)
    fields, geometry = read_snapshot(prefix.with_name(prefix.name+'.vtk'))
    q = np.asarray(fields['Q'][..., 0], dtype=np.float32)
    if not float(q.min()) < level < float(q.max()):
        raise ValueError('isosurface level lies outside the Q field')
    vertices, faces, _, _ = marching_cubes(q, level=level, step_size=step, allow_degenerate=False)
    omega = np.asarray(fields['vorticity'], dtype=np.float32)
    magnitude = np.sqrt(np.einsum('...i,...i->...', omega, omega))
    del omega
    vertex_color = map_coordinates(magnitude, vertices.T, order=1, mode='nearest')
    face_color_value = vertex_color[faces].mean(axis=1)
    vertices = vertices*np.array(geometry['spacing'])+np.array(geometry['origin'])
    polygons = vertices[faces]
    upper = float(np.quantile(vertex_color, .99))
    norm = colors.Normalize(vmin=0., vmax=upper, clip=True)
    cmap = colormaps['turbo']
    rgba = cmap(norm(face_color_value))
    normal = np.cross(polygons[:, 1]-polygons[:, 0], polygons[:, 2]-polygons[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    light = np.array([1., -1., 2.]); light /= np.linalg.norm(light)
    illumination = .45+.55*np.abs(normal@light)
    rgba[:, :3] *= illumination[:, None]
    fig = plt.figure(figsize=(9.4, 8.4), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    surface = Poly3DCollection(polygons, facecolors=rgba, edgecolors='none', linewidths=0, rasterized=True)
    ax.add_collection3d(surface)
    ax.set(xlim=(-math.pi, math.pi), ylim=(-math.pi, math.pi), zlim=(-math.pi, math.pi),
           xlabel=r'$x/L$', ylabel=r'$y/L$', zlabel=r'$z/L$')
    for setter in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        setter([-math.pi, 0., math.pi], [r'$-\pi$', '0', r'$\pi$'])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-56)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.97, .98, 1., 1.))
        axis.pane.set_edgecolor('.82')
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = fig.colorbar(scalar, ax=ax, shrink=.55, pad=.04, extend='max')
    bar.set_label(r'Vorticity magnitude $|\omega|L/U_0$')
    fig.suptitle('Taylor–Green vortex at t = 10', fontsize=16, y=.96)
    ax.set_title(rf'$Q L^2/U_0^2 = {level:g}$  |  $Re=1600$, $Ma=0.3$', fontsize=11, pad=9)
    fig.text(.5, .045, f'64³ elements, p=3  •  256³ sampled field  •  marching-cubes stride {step}\n'
             'Instantaneous structure visualization; threshold and color scale are explicitly fixed.',
             ha='center', fontsize=9, color='.35')
    fig.savefig(output/'vortex_structure_t10.png', dpi=200, bbox_inches='tight')
    fig.savefig(output/'vortex_structure_t10.pdf', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return dict(Q_level=level, marching_cubes_step=step, full_field_shape=list(q.shape),
                vertices=len(vertices), triangles=len(faces), color='vorticity magnitude',
                color_limits=[0., upper], color_upper_rule='99th percentile on isosurface vertices; clipped above',
                coordinates='sample-center coordinates from VTK origin and spacing',
                surface_note='marching-cubes stride affects visualization only; spectrum uses full 256^3 samples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--prefix', default='snapshot-010.0000')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--q-level', type=float, default=2.5)
    parser.add_argument('--surface-step', type=int, default=2)
    args = parser.parse_args()
    if args.surface_step < 1:
        parser.error('positive surface-step required')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fr-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    output = args.output or args.directory/'visualization'
    output.mkdir(parents=True, exist_ok=True)
    prefix = args.directory/args.prefix
    report = dict(spectrum=plot_spectra(prefix, output))
    print('Spectrum plotted', flush=True)
    report['vortices'] = render_vortices(prefix, output, level=args.q_level, step=args.surface_step)
    (output/'visualization.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
