"""Generate a Cartesian mesh and plot sections through a round jet."""
import argparse
import math
from pathlib import Path

import torch

from fr import CartesianMesh


DEFAULT_CELLS = (40, 32, 32)
NOZZLE_RADIUS = 0.5


def build_mesh(cells=DEFAULT_CELLS, *, length=16., half_width=4.,
               axial_stretch=2., transverse_stretch=3.3):
    """Return a mesh in units of the nozzle diameter."""
    if len(cells) != 3 or any(not isinstance(n, int) or n < 1 for n in cells):
        raise ValueError('three positive integer cell counts required')
    if (not math.isfinite(length) or length <= 0
            or not math.isfinite(half_width) or half_width <= NOZZLE_RADIUS):
        raise ValueError('positive finite length and half-width greater than 0.5 required')
    if any(not math.isfinite(s) or s < 0 for s in (axial_stretch, transverse_stretch)):
        raise ValueError('finite nonnegative stretch required')

    nx, ny, nz = cells
    xi = torch.arange(nx+1, dtype=torch.float64)/nx
    if axial_stretch:
        mapped = torch.expm1(axial_stretch*xi)
        xi = mapped/mapped[-1]
    edges = [length*xi]
    for n in (ny, nz):
        eta = (2*torch.arange(n+1, dtype=torch.float64)-n)/n
        if transverse_stretch:
            mapped = torch.sinh(transverse_stretch*eta)
            eta = mapped/mapped[-1]
        edges.append(half_width*eta)
    return CartesianMesh(edges)


def plot_mesh(mesh, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    def draw_grid(ax, horizontal, vertical):
        ax.vlines(horizontal, vertical[0], vertical[-1], color='#73889d', linewidth=.45)
        ax.hlines(vertical, horizontal[0], horizontal[-1], color='#73889d', linewidth=.45)
        ax.set(xlim=(horizontal[0], horizontal[-1]), ylim=(vertical[0], vertical[-1]),
               aspect='equal')

    x, y, z = (e.numpy() for e in mesh.edges)
    figure = Figure(figsize=(11, 7), layout='constrained')
    FigureCanvasAgg(figure)
    layout = figure.add_gridspec(2, 2)
    axial = figure.add_subplot(layout[0, :])
    inlet = figure.add_subplot(layout[1, 0])
    closeup = figure.add_subplot(layout[1, 1])

    draw_grid(axial, x, y)
    axial.axhline(0, color='#34495e', linestyle='--', linewidth=.8)
    axial.plot([0, 0], [-NOZZLE_RADIUS, NOZZLE_RADIUS], color='#be4c27',
               linewidth=2.5, clip_on=False)
    axial.set(title='Axial section (z/D = 0)', xlabel='x/D', ylabel='y/D')
    for ax in (inlet, closeup):
        draw_grid(ax, y, z)
        ax.add_patch(Circle((0, 0), NOZZLE_RADIUS, fill=False, color='#be4c27', linewidth=1.8))
        ax.set(xlabel='y/D', ylabel='z/D')
    inlet.set_title('Inlet plane (x/D = 0)')
    zoom = min(float(y[-1]), float(z[-1]), 1.)
    closeup.set(title='Inlet near the nozzle', xlim=(-zoom, zoom), ylim=(-zoom, zoom))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', nargs=3, type=int, default=DEFAULT_CELLS)
    parser.add_argument('--length', type=float, default=16.)
    parser.add_argument('--half-width', type=float, default=4.)
    parser.add_argument('--axial-stretch', type=float, default=2.)
    parser.add_argument('--transverse-stretch', type=float, default=3.3)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('mesh')/'cartesian.json')
    parser.add_argument('--plot', type=Path, default=Path('results/round_jet_les/mesh.png'))
    args = parser.parse_args()
    if args.output.resolve() == args.plot.resolve():
        parser.error('mesh and plot must use distinct output paths')
    mesh = build_mesh(tuple(args.cells), length=args.length, half_width=args.half_width,
                      axial_stretch=args.axial_stretch, transverse_stretch=args.transverse_stretch)
    mesh.write(args.output)
    plot_mesh(mesh, args.plot)
    print(f'{args.output}: {mesh.cells}, {math.prod(mesh.cells)} cells')
    for axis, edges in zip('xyz', mesh.edges):
        widths = edges.diff()
        print(f'{axis}: cell widths {widths.min():.4f}–{widths.max():.4f} D')
    print(args.plot)


if __name__ == '__main__':
    main()
