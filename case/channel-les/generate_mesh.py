"""Generate a channel LES mesh with symmetric tanh wall-normal stretching."""
import argparse
from pathlib import Path
from fr import CartesianMesh


DNS_MESH = Path(__file__).resolve().parents[1]/'channel-dns'/'mesh'/'default.json'


def build_mesh(cells=(32, 12, 16), stretch=1.2, source=DNS_MESH):
    """Keep the DNS physical bounds; cell counts are the final mesh counts."""
    dns = CartesianMesh.read(source)
    return CartesianMesh.rectangle(tuple(cells), dns.bounds, stretch=stretch)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cells', nargs=3, type=int, default=[32, 12, 16])
    parser.add_argument('--stretch', type=float, default=1.2)
    parser.add_argument('--source', type=Path, default=DNS_MESH)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('mesh')/'default.json')
    args = parser.parse_args()
    mesh = build_mesh(tuple(args.cells), args.stretch, args.source)
    mesh.write(args.output)
    widths = [e.diff() for e in mesh.edges]
    ratio = max(float(w.max()) for w in widths)/min(float(w.min()) for w in widths)
    print(f'{args.output}: cells={mesh.cells}, wall_dy={float(widths[1][0]):.9g}, '
          f'center_dy={float(widths[1].max()):.9g}, max_aspect={ratio:.6g}')


if __name__ == '__main__':
    main()
