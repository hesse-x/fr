"""Steady axisymmetric round-jet RANS in the thin shear layer approximation."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

from case.round_jet_les.compare import half_width
from case.round_jet_les.postprocess import write_csv


NOZZLE_RADIUS = 0.5
C_MU = 0.09
C_EPSILON_1 = 1.44
C_EPSILON_2 = 1.92
SIGMA_K = 1.0
SIGMA_EPSILON = 1.3
AMBIENT_K = 1e-9
DEFAULT_TURBULENCE_LENGTH = 0.005
DEFAULT_OUTPUT = Path('results/round_jet_rans')


@dataclass(frozen=True)
class RoundJetSolution:
    x: np.ndarray
    radial_faces: np.ndarray
    radius: np.ndarray
    axial_velocity: np.ndarray
    radial_flux: np.ndarray
    k: np.ndarray
    epsilon: np.ndarray
    nu_t: np.ndarray
    coflow: float
    mass_balance_error: float
    momentum_balance_error: float

    @property
    def radial_velocity(self):
        face_velocity = self.radial_flux[:, 1:]/self.radial_faces[1:]
        return 0.5*(np.pad(face_velocity, ((0, 0), (1, 0)))[:, :-1]+face_velocity)

    @property
    def centerline_velocity(self):
        r0, r1 = self.radius[:2]
        u0, u1 = self.axial_velocity[:, :2].T
        return (r1**2*u0-r0**2*u1)/(r1**2-r0**2)

    @property
    def half_velocity_radius(self):
        radius = np.r_[0., self.radius]
        profiles = np.column_stack((self.centerline_velocity, self.axial_velocity))
        return np.array([half_width(radius, row-self.coflow) for row in profiles])


def _face_flux(values, exterior, mass_flux, diffusion):
    """Outward radial transport across the right face of each cell."""
    right = np.r_[values[1:], exterior]
    upwind = np.where(mass_flux >= 0, values, right)
    return mass_flux*upwind + diffusion*(values-right)


def _divergence(face_flux):
    return np.diff(face_flux, prepend=0.)


def _mean_flow(previous, previous_flux, area_over_step, diffusion, coflow, x):
    """Solve continuity and excess axial momentum for cell U and face rV."""
    velocity = previous.copy()
    mass_flux = previous_flux.copy()
    n = velocity.size
    jacobian = np.zeros((5, 2*n))
    residual = np.empty(2*n)
    jacobian[3, 0::2] = area_over_step
    jacobian[4, 1:-1:2] = -1.

    for _ in range(25):
        excess = velocity-coflow
        momentum_flux = _face_flux(excess, 0., mass_flux, diffusion)
        residual[0::2] = area_over_step*(velocity*excess-previous*(previous-coflow))
        residual[0::2] += _divergence(momentum_flux)
        residual[1::2] = area_over_step*(velocity-previous)+_divergence(mass_flux)
        if np.max(np.abs(residual)) < 1e-12:
            return velocity, mass_flux, momentum_flux[-1]

        # Interleave U_i and (rV)_i so the momentum and continuity Jacobian is banded.
        left = diffusion+np.maximum(mass_flux, 0.)
        right = -diffusion+np.minimum(mass_flux, 0.)
        upwind = np.where(mass_flux >= 0, excess, np.r_[excess[1:], 0.])
        jacobian[0, 2::2] = right[:-1]
        jacobian[1, 1::2] = upwind
        jacobian[2, 0::2] = area_over_step*(2*velocity-coflow)+left
        jacobian[2, 2::2] -= right[:-1]
        jacobian[2, 1::2] = 1.
        jacobian[3, 1:-1:2] = -upwind[:-1]
        jacobian[4, :-2:2] = -left[:-1]
        update = solve_banded((2, 2), jacobian, -residual)
        velocity += update[0::2]
        mass_flux += update[1::2]
    raise RuntimeError(f'mean-flow Newton iteration did not converge at x/D={x:g}')


def _transport_scalar(previous, exterior, previous_weight, current_weight, mass_flux,
                      diffusion, source, sink, area):
    left = diffusion+np.maximum(mass_flux, 0.)
    right = -diffusion+np.minimum(mass_flux, 0.)
    matrix = np.zeros((3, previous.size))
    matrix[0, 1:] = right[:-1]
    matrix[1] = current_weight+left+area*sink
    matrix[1, 1:] -= right[:-1]
    matrix[2, :-1] = -left[:-1]
    rhs = previous_weight*previous+area*source
    rhs[-1] -= right[-1]*exterior
    return solve_banded((1, 1), matrix, rhs)


def solve_round_jet(*, length=25., radius=10., axial_steps=500, radial_cells=192, reynolds=6e5,
                    shear_width=0.08, turbulence_intensity=0.013,
                    turbulence_length=DEFAULT_TURBULENCE_LENGTH,
                    coflow=0.001):
    """March the mean velocity, turbulent kinetic energy and dissipation downstream."""
    parameters = (length, radius, reynolds, shear_width, turbulence_intensity,
                  turbulence_length, coflow)
    if not all(np.isfinite(value) and value > 0 for value in parameters):
        raise ValueError('physical and geometric parameters must be positive and finite')
    if radius <= NOZZLE_RADIUS or coflow >= 1:
        raise ValueError('radius must exceed the nozzle radius and coflow must be below jet speed')
    if not isinstance(axial_steps, int) or not isinstance(radial_cells, int):
        raise ValueError('grid counts must be integers')
    if axial_steps < 1 or radial_cells < 4:
        raise ValueError('at least one axial step and four radial cells are required')

    x = np.linspace(0., length, axial_steps+1)
    faces = radius*np.sinh(np.linspace(0., 3., radial_cells+1))/np.sinh(3.)
    r = 0.5*(faces[:-1]+faces[1:])
    area = 0.5*np.diff(faces**2)
    face_spacing = np.diff(np.r_[r, faces[-1]])
    area_over_step = area/(length/axial_steps)
    shape = (x.size, r.size)
    velocity = np.empty(shape)
    radial_flux = np.zeros((x.size, r.size+1))
    k, epsilon, nu_t = (np.empty(shape) for _ in range(3))
    weight = 0.5*(1-np.tanh((r-NOZZLE_RADIUS)/shear_width))
    velocity[0] = coflow+(1-coflow)*weight
    k[0] = AMBIENT_K+(1.5*turbulence_intensity**2-AMBIENT_K)*weight
    epsilon[0] = C_MU**0.75*k[0]**1.5/turbulence_length
    nu_t[0] = C_MU*k[0]**2/epsilon[0]
    ambient_epsilon = C_MU**0.75*AMBIENT_K**1.5/turbulence_length
    ambient_nu_t = C_MU*AMBIENT_K**2/ambient_epsilon
    mass_error = momentum_error = 0.

    for i in range(1, x.size):
        previous_u = velocity[i-1]
        viscosity = nu_t[i-1]
        # Linear interpolation of eddy viscosity to interior faces.
        face_viscosity = viscosity + (faces[1:]-r)/face_spacing * (
            np.r_[viscosity[1:], ambient_nu_t]-viscosity)
        molecular = faces[1:]/(reynolds*face_spacing)
        turbulent = faces[1:]*face_viscosity/face_spacing
        u, flux, edge_momentum_flux = _mean_flow(
            previous_u, radial_flux[i-1, 1:], area_over_step, molecular+turbulent, coflow, x[i])
        if np.any(u <= 0):
            raise RuntimeError(f'nonpositive axial velocity at x/D={x[i]:g}')
        velocity[i] = u
        radial_flux[i, 1:] = flux
        mass_error = max(mass_error, abs(area_over_step @ (u-previous_u)+flux[-1]))
        momentum_error = max(momentum_error, abs(
            area_over_step @ (u*(u-coflow)-previous_u*(previous_u-coflow))+edge_momentum_flux))

        shear = np.diff(np.r_[u, coflow])/face_spacing
        production = viscosity*(0.5*(shear+np.r_[0., shear[:-1]]))**2
        dissipation_rate = epsilon[i-1]/k[i-1]
        previous_weight, current_weight = area_over_step*previous_u, area_over_step*u
        k[i] = _transport_scalar(k[i-1], AMBIENT_K, previous_weight, current_weight, flux,
                                 molecular+turbulent/SIGMA_K, production, dissipation_rate, area)
        epsilon[i] = _transport_scalar(
            epsilon[i-1], ambient_epsilon, previous_weight, current_weight, flux,
            molecular+turbulent/SIGMA_EPSILON, C_EPSILON_1*production*dissipation_rate,
            C_EPSILON_2*dissipation_rate, area)
        if (not np.all(np.isfinite(k[i]+epsilon[i])) or np.any(k[i] <= 0)
                or np.any(epsilon[i] <= 0)):
            raise RuntimeError(f'invalid turbulence variables at x/D={x[i]:g}')
        nu_t[i] = C_MU*k[i]**2/epsilon[i]

    # The inlet sets axial variables only. Estimate its radial velocity one-sided.
    radial_flux[0] = radial_flux[1]
    return RoundJetSolution(x, faces, r, velocity, radial_flux, k, epsilon, nu_t,
                            coflow, mass_error, momentum_error)


def _compare_sp3(output, directory, x, centerline, width):
    def compare(filename, column, prediction):
        reference = np.genfromtxt(directory/filename, delimiter=',', names=True, ndmin=1)
        if not {'x_D', column}.issubset(reference.dtype.names or ()):
            raise ValueError(f'{filename} must contain x_D and {column}')
        stations, measured = reference['x_D'], reference[column]
        valid = (stations >= x[0]) & (stations <= x[-1]) & np.isfinite(measured)
        if not np.any(valid):
            raise ValueError(f'{filename} has no valid measurements within the axial domain')
        stations, measured = stations[valid], measured[valid]
        return stations, np.interp(stations, x, prediction), measured

    center = compare('centerline_vs_sp3.csv', 'sp3_ux_Uj', centerline)
    half_radius = compare('half_velocity_radius_vs_sp3.csv', 'sp3_r_half_D', width)
    if np.any(center[2] <= 0):
        raise ValueError('SP3 centerline speeds must be positive')
    write_csv(output/'centerline_vs_sp3.csv', ('x_D', 'rans_ux_Uj', 'sp3_ux_Uj'), zip(*center))
    write_csv(output/'half_velocity_radius_vs_sp3.csv',
              ('x_D', 'rans_r_half_D', 'sp3_r_half_D'), zip(*half_radius))
    errors = {
        'sp3_centerline_mape_percent': float(100*np.mean(np.abs(center[1]/center[2]-1))),
        'sp3_half_velocity_radius_rmse_D': float(np.sqrt(np.mean((half_radius[1]-half_radius[2])**2))),
    }
    return center, half_radius, errors


def write_results(solution, settings, output, *, plot=True, sp3_tables=None):
    output.mkdir(parents=True, exist_ok=True)
    x, r = solution.x, solution.radius
    centerline = solution.centerline_velocity
    width = solution.half_velocity_radius
    if not np.all(np.isfinite(width)):
        raise RuntimeError('the radial domain does not capture the half-velocity radius')
    area = np.pi*np.diff(solution.radial_faces**2)
    excess = solution.axial_velocity-solution.coflow
    excess_mass = excess @ area
    excess_momentum = (solution.axial_velocity*excess) @ area
    write_csv(output/'centerline.csv',
              ('x_D', 'ux_Uj', 'r_half_D', 'excess_volume_flux_Uj_D2',
               'excess_momentum_flux_Uj2_D2'),
              zip(x, centerline, width, excess_mass, excess_momentum))
    axial_grid, radial_grid = np.broadcast_arrays(x[:, None], r)
    columns = (axial_grid, radial_grid, solution.axial_velocity, solution.radial_velocity,
               solution.k, solution.epsilon, solution.nu_t)
    write_csv(output/'radial_profiles.csv',
              ('x_D', 'r_D', 'ux_Uj', 'ur_Uj', 'k_Uj2', 'epsilon_Uj3_D', 'nu_t_Uj_D'),
              zip(*(column.ravel() for column in columns)))
    core_crossing = np.flatnonzero(centerline <= 0.95)
    core_length = None
    if core_crossing.size:
        i = core_crossing[0]
        core_length = 0. if i == 0 else float(np.interp(
            0.95, centerline[i-1:i+1][::-1], x[i-1:i+1][::-1]))
    diagnostics = {
        'model': 'standard k-epsilon, axisymmetric thin shear layer',
        **settings,
        'centerline_outlet_ux_Uj': float(centerline[-1]),
        'outlet_r_half_D': float(width[-1]),
        'potential_core_length_D': core_length,
        'outlet_to_inlet_excess_momentum_ratio': float(excess_momentum[-1]/excess_momentum[0]),
        'max_section_mass_balance_error': solution.mass_balance_error,
        'max_section_momentum_balance_error': solution.momentum_balance_error,
    }
    if sp3_tables is not None:
        sp3_center, sp3_width, errors = _compare_sp3(output, sp3_tables, x, centerline, width)
        diagnostics.update(errors)
    (output/'summary.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+'\n')
    if plot:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 7), layout='constrained')
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(2, 2)
        field = fig.add_subplot(grid[0, :])
        cross_section = np.column_stack((solution.axial_velocity[:, ::-1], solution.axial_velocity))
        image = field.pcolormesh(x, np.r_[-r[::-1], r], cross_section.T, vmin=0., vmax=1.,
                                shading='auto', cmap='viridis')
        field.plot(x, width, 'w--', linewidth=0.9)
        field.plot(x, -width, 'w--', linewidth=0.9)
        field.set(xlabel='x/D', ylabel='y/D', xlim=(x[0], x[-1]),
                  ylim=(-min(r[-1], 1.8*width.max()), min(r[-1], 1.8*width.max())))
        fig.colorbar(image, ax=field, label='Mean axial velocity / Uj')
        ax = fig.add_subplot(grid[1, 0])
        ax.plot(x, centerline, label='RANS')
        if sp3_tables is not None:
            ax.plot(sp3_center[0], sp3_center[2], '--', label='NASA SP3 PIV')
            ax.legend()
        ax.set(xlabel='x/D', ylabel='Centerline velocity / Uj', xlim=(x[0], x[-1]))
        ax = fig.add_subplot(grid[1, 1])
        ax.plot(x, width)
        if sp3_tables is not None:
            ax.plot(sp3_width[0], sp3_width[2], 'o--', markersize=3)
        ax.set(xlabel='x/D', ylabel='Half-velocity radius / D', xlim=(x[0], x[-1]))
        fig.savefig(output/'mean_flow.png', dpi=160)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=float, default=25., help='axial domain length / D')
    parser.add_argument('--radius', type=float, default=10., help='outer boundary radius / D')
    parser.add_argument('--axial-steps', type=int, default=500)
    parser.add_argument('--radial-cells', type=int, default=192)
    parser.add_argument('--reynolds', type=float, default=6e5)
    parser.add_argument('--shear-width', type=float, default=0.08, help='nozzle tanh width / D')
    parser.add_argument('--turbulence-intensity', type=float, default=0.013)
    parser.add_argument('--turbulence-length', type=float, default=DEFAULT_TURBULENCE_LENGTH,
                        help='inlet length scale / D')
    parser.add_argument('--coflow', type=float, default=0.001, help='ambient axial velocity / Uj')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--sp3-tables', type=Path, help='directory with exported SP3 comparison CSVs')
    parser.add_argument('--no-plot', action='store_true')
    settings = vars(parser.parse_args())
    output = settings.pop('output')
    no_plot = settings.pop('no_plot')
    sp3_tables = settings.pop('sp3_tables')
    try:
        solution = solve_round_jet(**settings)
        diagnostics = write_results(solution, settings, output, plot=not no_plot, sp3_tables=sp3_tables)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    print(f'Saved {output} (outlet Uc/Uj={diagnostics["centerline_outlet_ux_Uj"]:.4f}, '
          f'r_half/D={diagnostics["outlet_r_half_D"]:.4f})')
