"""Run the documented Ban et al. (2026) Case 4 approximation and compare cuts."""
import argparse
from pathlib import Path
import subprocess
import sys

from .paper_compare import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', type=int, default=2)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--local-refinement', type=int, default=4)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--voltage-length', type=float, default=1/15)
    parser.add_argument('--outer-boundary', choices=('silver-muller', 'pml'), default='silver-muller')
    parser.add_argument('--warmup', type=int, default=12)
    parser.add_argument('--periods', type=int, default=4)
    parser.add_argument('--pml-thickness', type=float, default=.5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reuse', action='store_true', help='only postprocess an existing solution')
    parser.add_argument('--sphere', action='store_true', help='also compute full-sphere directivity')
    args = parser.parse_args()
    if args.refinement < 2:
        parser.error('this preset requires refinement >= 2 to resolve the 0.25-wavelength shell')
    out = args.output or Path(f'results/parabolic_antenna/paper_ban2026/voltage_local{args.local_refinement}')
    if not args.reuse:
        subprocess.run([sys.executable, '-m', 'case.parabolic_antenna.solve',
            '--diameter', '2', '--focal-length', '.5', '--thickness', '.25', '--radius', '.12',
            '--refinement', str(args.refinement), '--threads', str(args.threads),
            '--local-refinement', str(args.local_refinement), '--p', str(args.p),
            '--feed', 'voltage', '--voltage-length', str(args.voltage_length),
            '--domain-half-width', '1.5', '--outer-boundary', args.outer_boundary,
            '--warmup', str(args.warmup), '--periods', str(args.periods),
            '--pml-thickness', str(args.pml_thickness), '--frames', '0', '--validate',
            '--output', str(out)], check=True)
    compare(out/'solution.pt', sphere=args.sphere)


if __name__ == '__main__':
    main()
