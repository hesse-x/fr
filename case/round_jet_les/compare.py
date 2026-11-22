"""Compare time-averaged jet velocity with the NASA SP3 consensus PIV data."""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .postprocess import write_csv


def read_sp3(archive, grid):
    with archive.open(f'Consensus1/1ptn/Tx_X_SP3_{grid}.dat') as stream:
        return np.loadtxt(stream, skiprows=2 if grid == '1ptn_Sym' else 5,
                          comments='AUXDATA', usecols=(0, 1, 3, 12))


def fold_profile(table):
    table = table.copy()
    table[np.abs(table[:, 1]) < 1e-6, 1] = 0.
    positive = table[table[:, 1] >= 0]
    negative = table[table[:, 1] <= 0][::-1]
    radius = positive[:, 1]
    if not np.array_equal(radius, -negative[:, 1]):
        raise ValueError('expected symmetric PIV radial coordinates')
    valid = (positive[:, 3] > 0) & (negative[:, 3] > 0)
    return radius[valid], (positive[valid, 2]+negative[valid, 2])/2


def half_width(radius, velocity):
    if not len(radius) or radius[0] != 0:
        return np.nan
    threshold = velocity[0]/2
    crossing = np.flatnonzero(velocity[1:] <= threshold)
    if not crossing.size:
        return np.nan
    i = crossing[0]
    return np.interp(threshold, velocity[i:i+2][::-1], radius[i:i+2][::-1])


def compare(run, reference, output, *, expected_steps=None):
    summary = json.loads((run/'summary.json').read_text())
    if not summary['samples']:
        raise ValueError('PIV comparison requires time-averaged LES profiles')
    if expected_steps is not None and summary['steps'] != expected_steps:
        raise ValueError(f'expected {expected_steps} steps, found {summary["steps"]}')

    center = np.loadtxt(run/'centerline.csv', delimiter=',', skiprows=1, ndmin=2)
    radial = np.loadtxt(run/'radial_profiles.csv', delimiter=',', skiprows=1, ndmin=2)
    with zipfile.ZipFile(reference) as archive:
        piv_axial = read_sp3(archive, 'SimpleX')
        piv_plane = read_sp3(archive, '1ptn_Sym')
    piv_center = piv_axial[piv_axial[:, 1] == 0]
    x_les = center[:, 0]
    piv_center = piv_center[(piv_center[:, 0] >= x_les[0]) & (piv_center[:, 0] <= x_les[-1])]
    if not len(piv_center):
        raise ValueError('LES and SP3 axial grids do not overlap')
    x = piv_center[:, 0]
    u_piv = piv_center[:, 2]
    u_les = np.interp(x, x_les, center[:, 1])
    w_piv = np.array([half_width(*fold_profile(piv_plane[piv_plane[:, 0] == station]))
                      for station in x])
    w_les = np.interp(x, x_les, center[:, 2])

    fig = Figure(figsize=(10, 8), layout='constrained')
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 6)
    axial = fig.add_subplot(grid[0, :3])
    widths = fig.add_subplot(grid[0, 3:])
    axial.plot(x, u_les, label='LES')
    axial.plot(x, u_piv, '--', label='NASA SP3 PIV')
    axial.set(xlabel='x/D', ylabel='Centerline mean axial velocity / Uj')
    axial.legend()
    widths.plot(x, w_les)
    widths.plot(x, w_piv, '--')
    widths.set(xlabel='x/D', ylabel='Half-velocity radius / D')

    stations = np.unique(radial[:, 0])
    if len(stations) > 6:
        raise ValueError('expected at most six LES radial stations')
    radial_rows = []
    for i, station in enumerate(stations):
        data = piv_plane[np.isclose(piv_plane[:, 0], station, rtol=0, atol=1e-8)]
        if not len(data):
            continue
        profile = radial[radial[:, 0] == station]
        r, measured = fold_profile(data)
        overlap = r <= profile[-1, 1]
        r, measured = r[overlap], measured[overlap]
        les = np.interp(r, profile[:, 1], profile[:, 2])
        radial_rows.extend(zip(np.full_like(r, station), r, les, measured))
        ax = fig.add_subplot(grid[1+i//3, 2*(i % 3):2*(i % 3)+2])
        ax.plot(r, les)
        ax.plot(r, measured, '--')
        ax.set(title=f'x/D = {station:g}', xlabel='r/D', ylabel='Mean axial velocity / Uj')
    last_sample_time = summary.get('last_sample_time', summary['time_Uj_D'])
    fig.suptitle(f'Round jet vs NASA SP3  |  {summary["steps"]:,} steps, '
                 f'{summary["samples"]:,} samples, t Uj/D = '
                 f'{summary["sample_from"]:g}–{last_sample_time:.3g}')

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'piv_centerline.csv',
              ('x_D', 'u_LES_Uj', 'u_PIV_Uid', 'half_width_LES_D', 'half_width_PIV_D'),
              zip(x, u_les, u_piv, w_les, w_piv))
    write_csv(output/'piv_radial_profiles.csv', ('x_D', 'r_D', 'u_LES_Uj', 'u_PIV_Uid'),
              radial_rows)
    fig.savefig(output/'piv_comparison.png', dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='directory containing LES profile CSVs')
    parser.add_argument('reference', type=Path, help='official NASA consensus1.zip archive')
    parser.add_argument('--output', type=Path, help='destination for the comparison files')
    parser.add_argument('--expected-steps', type=int)
    args = parser.parse_args()
    compare(args.run, args.reference, args.output or args.run, expected_steps=args.expected_steps)


if __name__ == '__main__':
    main()
