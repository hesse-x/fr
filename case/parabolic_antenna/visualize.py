"""Plot native results, or regenerate figures from a saved solution.pt."""
import argparse
from pathlib import Path

import torch


def field_plot(case, physical, out, time, plane='xy'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, SymLogNorm
    from matplotlib.patches import Rectangle
    from fr.plotting import FieldSampler

    horizontal, normal = (0, 2) if plane == 'xy' else (2, 0)
    if plane == 'xy':
        sampler = FieldSampler(case)
        fields = sampler.state(physical)
    else:
        from types import SimpleNamespace
        order = (2, 1, 0)
        proxy = SimpleNamespace(dim=3, p=case.p, z=case.z, device=case.device,
            cells=tuple(case.cells[a] for a in order), bounds=tuple(case.bounds[a] for a in order),
            edges=[case.edges[a] for a in order], widths=tuple(case.widths[a] for a in order))
        sampler = FieldSampler(proxy)
        fields = sampler.state(physical.permute(2, 1, 0, 5, 4, 3, 6))
    ez = fields[:, 2].numpy()
    magnitude = fields[:, :3].norm(dim=-1).numpy()
    # Limit the display range using the radiating region, so singular source
    # near-fields do not hide the waves. The saved solution is unchanged.
    centers = sampler.polygons.mean(1)
    visible = centers[:, 0]**2+(centers[:, 1]-case.focal_length)**2 >= .25**2
    peak = max(float(abs(ez[visible]).max()), 1e-30)
    emax = max(float(magnitude[visible].max()), 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, values, norm, cmap, title, label in zip(
        axes, (ez, magnitude),
        (SymLogNorm(linthresh=peak*.001, vmin=-peak, vmax=peak), LogNorm(vmin=emax*1e-5, vmax=emax)),
        ('RdBu_r', 'inferno'), ('Instantaneous Ez', 'Instantaneous |E|'), ('Ez (symmetric log scale)', '|E| (log scale)')):
        artist = PolyCollection(sampler.polygons, array=values, norm=norm, cmap=cmap,
                                edgecolors='none', antialiased=False, rasterized=True)
        ax.add_collection(artist)
        # Draw the actual stair-step PEC cells in the same z=0 plane.
        normal_cell = sampler.zslice[0]
        solid = case.solid[:, :, normal_cell] if plane == 'xy' else case.solid[normal_cell, :, :].T
        for i, j in solid.nonzero().tolist():
            ax.add_patch(Rectangle((float(case.edges[horizontal][i]), float(case.edges[1][j])),
                                  float(case.widths[horizontal][i]), float(case.widths[1][j]), color='.55', zorder=3))
        ax.plot([], [], color='.55', lw=7, label='PEC reflector')
        ax.plot(0., case.focal_length, '*', color='limegreen', markeredgecolor='black',
                markersize=10, label='Feed focus', zorder=4)
        x0, x1 = case.pml.inner_bounds[horizontal]
        y0, y1 = case.pml.inner_bounds[1]
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                              linestyle='--', edgecolor='cyan', label='PML entrance' if case.outer_boundary == 'pml' else 'Silver-Muller ABC'))
        ax.set(xlim=case.bounds[horizontal], ylim=case.bounds[1], xlabel=('x' if plane == 'xy' else 'z')+' / wavelength',
               ylabel='y / wavelength', title=title)
        ax.set_aspect('equal')
        fig.colorbar(artist, ax=ax, shrink=.75, label=label, extend='both')
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle(f'Parabolic antenna: full computational domain, {plane} slice '+('z=0' if plane == 'xy' else 'x=0')+f', t={time:g}\nDisplay range from r >= 0.25 wavelength about feed; source peaks saturate')
    fig.savefig(Path(out)/('field.png' if plane == 'xy' else 'field_E.png'), dpi=180)
    plt.close(fig)
    if plane == 'xy':
        field_plot(case, physical, out, time, plane='yz')
        if case.nonuniform:
            mesh_plot(case, out)


def mesh_plot(case, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for z in case.edges[2]:
        ax.axvline(float(z), color='.8', lw=.4, zorder=0)
    for y in case.edges[1]:
        ax.axhline(float(y), color='.8', lw=.4, zorder=0)
    xcell = int(torch.searchsorted(case.edges[0], torch.tensor(0., dtype=torch.float64), right=True))-1
    for j, k in case.solid[xcell].nonzero().tolist():
        ax.add_patch(Rectangle((float(case.edges[2][k]), float(case.edges[1][j])),
            float(case.widths[2][k]), float(case.widths[1][j]), color='.5'))
    z = torch.linspace(-case.diameter/2, case.diameter/2, 201)
    ax.plot(z, z*z/(4*case.focal_length), '--', color='black', lw=1., label='Ideal PEC surface')
    if case.feed == 'voltage':
        ax.plot([-case.voltage_length/2, case.voltage_length/2], [case.focal_length]*2,
                color='red', lw=3, label='Prescribed voltage edge')
        zoom = ax.inset_axes([.65, .67, .32, .28])
        for zedge in case.edges[2]: zoom.axvline(float(zedge), color='.6', lw=.6)
        for yedge in case.edges[1]: zoom.axhline(float(yedge), color='.6', lw=.6)
        zoom.plot([-case.voltage_length/2, case.voltage_length/2], [case.focal_length]*2, 'r-', lw=3)
        zoom.set(xlim=(-.12, .12), ylim=(case.focal_length-.12, case.focal_length+.12), title='Voltage-edge detail')
        zoom.tick_params(labelsize=7)
    ax.set(xlim=case.bounds[2], ylim=case.bounds[1], xlabel='z / wavelength', ylabel='y / wavelength',
           title=f'Locally graded mesh, yz slice at x=0\n{case.cells[0]} x {case.cells[1]} x {case.cells[2]} elements')
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=8)
    fig.savefig(Path(out)/'mesh.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('solution', type=Path)
    args = parser.parse_args()
    # This command opens a locally generated solver artifact.
    data = torch.load(args.solution, map_location='cpu', weights_only=True)
    stats = data['stats']
    torch.set_num_threads(1)
    from .solve import ParabolicAntennaCase
    if 'edges' in data:
        h = max(float((e[1:]-e[:-1]).max()) for e in data['edges'])
    else:
        x = data['coords'][0]
        h = float(x[1, 0, 0, 0, 0, 0]-x[0, 0, 0, 0, 0, 0])
    case = ParabolicAntennaCase(diameter=stats['diameter'], focal_length=stats['focal_length'],
        thickness=stats['reflector_thickness'], radius=stats.get('radius', .3),
        refinement=round(.5/h), p=stats['p'], pml_thickness=stats['pml_thickness'] or .5,
        local_refinement=stats.get('local_refinement', 1), feed=stats.get('feed', 'current'),
        voltage_length=stats.get('voltage_length') or 1/15, voltage_amplitude=stats.get('voltage_amplitude') or 1.,
        domain_half_width=stats.get('domain_half_width'), outer_boundary=stats.get('outer_boundary', 'pml'))
    field_plot(case, data['U'], args.solution.parent, stats['time'])


if __name__ == '__main__':
    main()
