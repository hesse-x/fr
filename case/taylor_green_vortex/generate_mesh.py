"""Generate explicit Cartesian cell edges; run from the repository root."""
import argparse
import math
from pathlib import Path
from fr import CartesianMesh


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cells', nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('mesh')/'default.json')
    parser.add_argument('--stretch', type=float, default=0.0)
    args = parser.parse_args()
    CartesianMesh.rectangle(tuple(args.cells), ((-math.pi, math.pi),)*3, stretch=args.stretch).write(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
