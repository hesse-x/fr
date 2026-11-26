"""Plot the 0.08D round-jet RANS and LES results against SP3 PIV."""

import argparse
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


LES_TABLES = Path(__file__).resolve().parents[1]/'round_jet_les'/'validation'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='RANS output directory for the 0.08D inlet')
    parser.add_argument('--les-tables', type=Path, default=LES_TABLES)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    rans = np.genfromtxt(args.run/'centerline.csv', delimiter=',', names=True)
    fig = Figure(figsize=(9, 6), layout='constrained')
    FigureCanvasAgg(fig)
    speed, width = fig.subplots(2, 1, sharex=True)
    for ax, table, column, name in (
        (speed, 'centerline_vs_sp3.csv', 'ux_Uj', r'Centerline velocity $U_c/U_j$'),
        (width, 'half_velocity_radius_vs_sp3.csv', 'r_half_D', r'Half-velocity radius $r_{1/2}/D$'),
    ):
        data = np.genfromtxt(args.les_tables/table, delimiter=',', names=True)
        ax.plot(rans['x_D'], rans[column], color='#b54e20', label='RANS')
        ax.plot(data['x_D'], data[f'les_008_{column}'], color='#1769a3', label='LES')
        ax.plot(data['x_D'], data[f'sp3_{column}'], 'o', color='0.22',
                markersize=2.5, label='NASA SP3 PIV', zorder=3)
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
    speed.legend(frameon=False, ncols=3)
    width.set(xlabel=r'$x/D$', xlim=(0, rans['x_D'][-1]))
    output = args.output or args.run/'les_rans_sp3.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165)


if __name__ == '__main__':
    main()
