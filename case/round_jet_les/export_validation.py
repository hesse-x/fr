"""Export compact SMC000 LES and SP3 PIV comparisons for validation."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from .compare import fold_profile, half_width
from .mapped import ExtrudedProbes, MappedRoundJetLES


WIDTHS = (0.08, 0.10, 0.12)
LABELS = ('les_008', 'les_010', 'les_012', 'sp3')
SECTIONS = (5, 8, 10, 12, 16, 18)
STATIONS = np.arange(26, dtype=np.float64)
RADIUS = np.linspace(0, 5, 251)
SIMILARITY_RADIUS = np.linspace(0, 2.5, 101)
ANGLES = torch.arange(192, dtype=torch.float64) * (2 * torch.pi / 192)


def read_piv(archive, grid):
    with archive.open(f'Consensus1/1ptn/Tx_X_SP3_{grid}.dat') as stream:
        # x, y, U, V, uu, vv, ww, uv, and data quality.
        return np.loadtxt(stream, skiprows=2 if grid == '1ptn_Sym' else 5,
                          comments='AUXDATA', usecols=(0, 1, 3, 4, 6, 7, 8, 9, 12))


def piv_section(table):
    y = table[:, 1]
    radial_sign = np.sign(np.where(np.abs(y) < 1e-6, 0., y))
    k = .5 * table[:, 4:7].sum(axis=1)
    values = (table[:, 2], radial_sign * table[:, 3], k,
              radial_sign * table[:, 7], table[:, 4])
    columns = []
    for value in values:
        r, folded = fold_profile(np.column_stack((table[:, :2], value, table[:, -1])))
        columns.append(folded)
    columns[-1] = np.sqrt(np.maximum(columns[-1], 0.))
    return r, np.column_stack(columns)


def sample_rings(probe, axial, radius, fields):
    x, r, theta = torch.meshgrid(torch.as_tensor(axial), torch.as_tensor(radius),
                                 ANGLES, indexing='ij')
    evaluate = probe((x.flatten(), (r * theta.cos()).flatten(), (r * theta.sin()).flatten()))
    return [evaluate(field).reshape(x.shape).numpy() for field in fields]


def les_data(directory, width, model, x_piv, sections):
    summary = json.loads((directory/'summary.json').read_text())
    data = torch.load(directory/'field_statistics.pt', weights_only=True, mmap=True, map_location='cpu')
    if (summary['shear_width_D'] != width or data['mesh_sha256'] != model.mesh_digest
            or data['samples'] != summary['samples'] or data['steps'] != summary['steps']
            or summary['profiles'] != 'time mean'):
        raise ValueError(f'incompatible mean-field data in {directory}')
    probe = ExtrudedProbes(model.edges, model.section, np.argsort(data['mesh_cell_order'].numpy()),
                           model.s.z)
    u, v, w = data['velocity_mean_Uj']
    uu, _, _, uv, uw, _ = data['reynolds_stress_Uj2']
    density, k = data['density_mean_rho_inf'], data['resolved_tke_Uj2']

    x, uc, _ = np.loadtxt(directory/'centerline.csv', delimiter=',', skiprows=1).T
    axial = torch.as_tensor(x_piv.copy())
    zero = torch.zeros_like(axial)
    on_axis = probe((axial, zero, zero))
    center = np.column_stack((np.interp(x_piv, x, uc), on_axis(k).numpy(),
                              np.sqrt(on_axis(uu).numpy().clip(min=0.))))

    radial = []
    cos, sin = ANGLES.cos().numpy(), ANGLES.sin().numpy()
    for station, radius, _ in sections:
        ux, uy, uz, k_values, uv_values, uw_values, uu_values = sample_rings(
            probe, np.array([station]), radius, (u, v, w, k, uv, uw, uu))
        radial.append(np.column_stack((
            ux[0].mean(axis=-1),
            (uy[0]*cos + uz[0]*sin).mean(axis=-1),
            k_values[0].mean(axis=-1),
            (uv_values[0]*cos + uw_values[0]*sin).mean(axis=-1),
            np.sqrt(uu_values[0].mean(axis=-1).clip(min=0.)),
        )))

    bands = []
    for rings in np.array_split(RADIUS, 16):
        velocity, rho, variance = sample_rings(probe, STATIONS, rings, (u, density, uu))
        mass = rho*velocity
        momentum = mass*velocity
        bands.append(np.stack((velocity.mean(axis=-1), mass.mean(axis=-1),
                               momentum.mean(axis=-1), (momentum + rho*variance).mean(axis=-1))))
    profiles, mass, momentum, momentum_with_variance = np.concatenate(bands, axis=-1)
    np.testing.assert_allclose(profiles[:, 0], np.interp(STATIONS, x, uc), atol=5e-12)
    widths = np.array([half_width(RADIUS, row) for row in profiles])
    if not np.isfinite(widths).all() or np.any(widths*SIMILARITY_RADIUS[-1] > RADIUS[-1]):
        raise ValueError(f'radial grid does not cover similarity profiles in {directory}')
    similarity = np.array([np.interp(SIMILARITY_RADIUS, RADIUS/h, row/row[0])
                           for h, row in zip(widths, profiles)])
    fluxes = np.column_stack([
        2*np.pi*np.trapezoid(values*RADIUS, RADIUS, axis=-1)
        for values in (mass, momentum, momentum_with_variance)
    ])
    return center, np.concatenate(radial), widths, similarity, fluxes


def export(runs, reference, output):
    with ZipFile(reference) as archive:
        piv_axis = read_piv(archive, 'SimpleX')
        piv_plane = read_piv(archive, '1ptn_Sym')
    piv_axis = piv_axis[(np.abs(piv_axis[:, 1]) < 1e-8) & (piv_axis[:, -1] > 0)
                        & (piv_axis[:, 0] >= 1) & (piv_axis[:, 0] <= 25)]
    piv_axis = piv_axis[np.argsort(piv_axis[:, 0])]
    centers = np.column_stack((piv_axis[:, 2], .5*piv_axis[:, 4:7].sum(axis=1),
                               np.sqrt(piv_axis[:, 4].clip(min=0.))))

    sections, exp_widths = [], []
    for station in STATIONS:
        r, values = piv_section(piv_plane[piv_plane[:, 0] == station])
        exp_widths.append(half_width(r, values[:, 0]))
        if station in SECTIONS:
            if not len(r) or r[0] != 0:
                raise ValueError(f'PIV section at x/D={station:g} does not reach the axis')
            sections.append((station, r, values))
    piv_radial = np.concatenate([values for _, _, values in sections])
    radial_coords = np.concatenate([np.column_stack((np.full_like(r, x), r)) for x, r, _ in sections])

    summary = json.loads((runs[0]/'summary.json').read_text())
    model = MappedRoundJetLES(
        runs[0]/'mesh.msh', reynolds=summary['reynolds_D'], mach=summary['mach'],
        shear_width=summary['shear_width_D'], disturbance=summary['disturbance'],
        modal_filter=summary['modal_filter'], outlet=summary['outlet'],
    )
    center, radial, width, similarity, flux = zip(*[
        les_data(path, w, model, piv_axis[:, 0], sections) for path, w in zip(runs, WIDTHS)
    ])

    def save(name, coords, values, quantities, labels=LABELS):
        columns = [f'{label}_{quantity}' for label in labels for quantity in quantities]
        np.savetxt(output/name, np.column_stack((*coords.values(), *values)), delimiter=',',
                   header=','.join((*coords.keys(), *columns)), comments='', fmt='%.9g')

    output.mkdir(parents=True, exist_ok=True)
    save('centerline_vs_sp3.csv', {'x_D': piv_axis[:, 0]}, (*center, centers),
         ('ux_Uj', 'k_Uj2', 'ux_rms_Uj'))
    save('radial_profiles_vs_sp3.csv', {'x_D': radial_coords[:, 0], 'r_D': radial_coords[:, 1]},
         (*radial, piv_radial), ('ux_Uj', 'ur_Uj', 'k_Uj2', 'uxur_Uj2', 'ux_rms_Uj'))
    save('half_velocity_radius_vs_sp3.csv', {'x_D': STATIONS}, (*width, exp_widths), ('r_half_D',))
    save('velocity_similarity.csv', {
        'x_D': np.repeat(STATIONS, len(SIMILARITY_RADIUS)),
        'r_over_r_half': np.tile(SIMILARITY_RADIUS, len(STATIONS)),
    }, tuple(values.ravel() for values in similarity), ('azimuthal_mean_ux_Uc',), LABELS[:3])
    save('section_flux_diagnostics.csv', {'x_D': STATIONS}, flux,
         ('mass_mean', 'momentum_mean', 'momentum_mean_plus_resolved_variance'), LABELS[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, help='NASA SP3 consensus1.zip')
    parser.add_argument('--width-008', type=Path, required=True)
    parser.add_argument('--width-010', type=Path, required=True)
    parser.add_argument('--width-012', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('validation'))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    export((args.width_008, args.width_010, args.width_012), args.reference, args.output)


if __name__ == '__main__':
    main()
