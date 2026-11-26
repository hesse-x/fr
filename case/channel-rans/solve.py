"""Steady incompressible channel RANS with Wilcox2006-LRN."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root

from fr import CartesianMesh
from fr.operators import operators


DNS_CASE = Path(__file__).resolve().parents[1]/'channel-dns'
DNS_MESH = DNS_CASE/'mesh'/'default.json'
DNS_REFERENCE = DNS_CASE/'reference'/'chan180.means'
DNS_RE_TAU = 178.12
BETA_0 = 0.0708
BETA_STAR_0 = 0.09
SIGMA_K = 0.6
SIGMA_OMEGA = 0.5
C_LIM = 7/8


@dataclass(frozen=True)
class WilcoxDamping:
    r_k: float = 6.
    r_beta: float = 8.
    r_omega: float = 2.61

    def __post_init__(self):
        if not all(np.isfinite(value) and value > 0
                   for value in (self.r_k, self.r_beta, self.r_omega)):
            raise ValueError('Wilcox damping parameters must be positive and finite')


STANDARD_DAMPING = WilcoxDamping()


@dataclass(frozen=True)
class ChannelSolution:
    y_over_h: np.ndarray
    u_plus: np.ndarray
    k_plus: np.ndarray
    omega_plus: np.ndarray
    nu_t_over_nu: np.ndarray
    viscous_shear_plus: np.ndarray
    reynolds_shear_plus: np.ndarray
    center_u_plus: float
    bulk_u_plus: float
    residual: float
    evaluations: int


def half_channel_nodes(mesh, order):
    """Map the repository's Gauss nodes onto the lower half of the DNS mesh."""
    if len(mesh.edges) != 3 or not isinstance(order, int) or order < 0:
        raise ValueError('expected a 3D Cartesian mesh and a nonnegative Gauss polynomial order')
    y = mesh.edges[1].numpy()
    half_height = (y[-1]-y[0])/2
    scaled = (y-y[0])/half_height
    middle = len(y)//2
    if not np.allclose(scaled+scaled[::-1], 2., rtol=0, atol=1e-12) or not np.isclose(
            scaled[middle], 1., rtol=0, atol=1e-12):
        raise ValueError('wall-normal mesh must be symmetric with a face at the centerline')
    edges = scaled[:middle+1].copy()
    edges[-1] = 1.
    nodes = operators(order, 'cpu')[0].numpy()
    points = edges[:-1, None] + np.diff(edges)[:, None]*(nodes+1)/2
    if points.size < 2:
        raise ValueError('at least two Gauss points are required on the half-channel')
    return points.ravel()


def wilcox_terms(k, omega, nu, total_shear, damping=STANDARD_DAMPING):
    """Local damping, production and shear for the low-Reynolds-number model."""
    re_t = k/(nu*omega)
    re_k = re_t/damping.r_k
    re_omega = re_t/damping.r_omega
    alpha_star = (BETA_0/3 + re_k)/(1 + re_k)
    beta_star = BETA_STAR_0 * (1 - (1-100*BETA_0/27)/(1 + (re_t/damping.r_beta)**4))
    gamma = (13/25) * (1/9 + re_omega)/(1 + re_omega)/alpha_star
    diffusion_viscosity = alpha_star*k/omega
    limiter = C_LIM*np.sqrt(alpha_star/BETA_STAR_0)

    # In plane shear the stress-limited branch contributes alpha_star*k/limiter
    # to the shear stress, so the momentum relation can be inverted explicitly.
    unlimited = total_shear/(nu+diffusion_viscosity)
    limited = (total_shear-alpha_star*k/limiter)/nu
    gradient = np.maximum(unlimited, limited)
    effective_omega = np.maximum(omega, limiter*gradient)
    nu_t = alpha_star*k/effective_omega
    omega_production = gamma*omega*alpha_star/effective_omega*gradient**2
    return gradient, nu_t, diffusion_viscosity, beta_star, omega_production


def solve_channel(mesh, *, re_tau=DNS_RE_TAU, order=3, damping=STANDARD_DAMPING,
                  initial_solution=None):
    """Solve k and omega directly in wall units on a pressure-driven half-channel."""
    if not np.isfinite(re_tau) or re_tau <= 0:
        raise ValueError('Re_tau must be positive and finite')
    y = half_channel_nodes(mesh, order)
    nu = 1/re_tau
    # Length is scaled by h, velocity by u_tau, and the driving acceleration is 1.
    total_shear = 1-y
    dx = np.diff(y)
    faces = np.r_[0., (y[:-1]+y[1:])/2, 1.]
    volumes = np.diff(faces)
    left_weight = (faces[1:]-y)/volumes
    omega_coefficient = 6*nu/BETA_0
    omega_singular = omega_coefficient/y**2
    singular_gradient = -2*omega_coefficient/y**3
    singular_face_gradient = -2*omega_coefficient/faces[1:-1]**3
    wall_weights = np.array([y[1]/y[0], -y[0]/y[1]])/dx[0]

    def face_gradient(value, outer_gradient):
        return np.r_[wall_weights @ value[:2], np.diff(value)/dx, outer_gradient]

    def residual(log_state):
        k, omega = np.exp(log_state.reshape(-1, 2)).T
        gradient, nu_t, diffusion, beta_star, omega_production = wilcox_terms(
            k, omega, nu, total_shear, damping)
        regular = omega-omega_singular
        k_faces = face_gradient(k, 0.)
        regular_faces = face_gradient(regular, 2*omega_coefficient)
        diffusion_faces = (diffusion[1:]+diffusion[:-1])/2
        k_flux = nu*k_faces
        k_flux[1:-1] += SIGMA_K*diffusion_faces*k_faces[1:-1]

        # Subtract the analytic molecular-diffusion/destruction balance of
        # omega_singular before discretizing the regular remainder.
        omega_flux = nu*regular_faces
        omega_flux[1:-1] += SIGMA_OMEGA*diffusion_faces*(
            regular_faces[1:-1]+singular_face_gradient)
        dk = left_weight*k_faces[:-1]+(1-left_weight)*k_faces[1:]
        domega = singular_gradient + (
            left_weight*regular_faces[:-1]+(1-left_weight)*regular_faces[1:])
        cross_diffusion = np.maximum(dk*domega, 0.)/(8*omega)
        k_sink = beta_star*k*omega
        omega_sink = BETA_0*regular*(omega+omega_singular)
        k_residual = np.diff(k_flux)/volumes + nu_t*gradient**2 - k_sink
        omega_residual = np.diff(omega_flux)/volumes + omega_production - omega_sink + cross_diffusion
        return np.column_stack((k_residual/k_sink, omega_residual/(BETA_0*omega**2))).ravel()

    if initial_solution is None:
        k_start = 4*np.tanh(y*re_tau/18)**2*(1-.6*y**2)
        omega_start = omega_singular + 1/(.12*y)
    else:
        if not np.array_equal(initial_solution.y_over_h, y):
            raise ValueError('initial solution must use the same wall-normal nodes')
        k_start = initial_solution.k_plus
        omega_start = initial_solution.omega_plus/nu
    initial = np.log(np.column_stack((k_start, omega_start))).ravel()
    solution = root(residual, initial, method='hybr',
                    options={'band': (3, 3), 'factor': 0.1, 'xtol': 1e-10})
    error = float(np.max(np.abs(residual(solution.x))))
    if not np.isfinite(error) or error > 1e-8:
        raise RuntimeError(f'Wilcox solve did not converge (scaled residual {error:.3g}): {solution.message}')
    k, omega = np.exp(solution.x.reshape(-1, 2)).T
    gradient, nu_t, *_ = wilcox_terms(k, omega, nu, total_shear, damping)

    # The prescribed total shear vanishes at the center and is entirely
    # molecular at the wall. Integrating dU/dy with U(0)=0 gives U and U_bulk.
    coordinates = np.r_[0., y, 1.]
    u_integral = PchipInterpolator(coordinates, np.r_[re_tau, gradient, 0.]).antiderivative()
    bulk_integral = u_integral.antiderivative()
    wall = u_integral(0.)
    return ChannelSolution(
        y, u_integral(y)-wall, k, omega*nu, nu_t/nu, nu*gradient, nu_t*gradient,
        float(u_integral(1.)-wall), float(bulk_integral(1.)-bulk_integral(0.)-wall),
        error, solution.nfev,
    )


def write_results(solution, mesh, mesh_path, order, re_tau, output):
    output.mkdir(parents=True, exist_ok=True)
    y, u = solution.y_over_h, solution.u_plus
    y_profile, u_profile = np.r_[0., y, 1.], np.r_[0., u, solution.center_u_plus]
    dns_y, dns_y_plus, dns_u, dns_gradient = np.loadtxt(DNS_REFERENCE, usecols=(0, 1, 2, 3)).T
    dns_bulk = float(PchipInterpolator(dns_y, dns_u).integrate(0., 1.))
    comparison = np.interp(dns_y, y_profile, u_profile)
    rms = float(np.sqrt(np.mean((comparison-dns_u)**2)))
    summary = {
        'model': 'Wilcox2006-LRN',
        're_tau': re_tau,
        'mesh': str(mesh_path),
        'mesh_cells': mesh.cells,
        'gauss_order': order,
        'half_channel_points': len(y),
        'first_y_plus': float(y[0]*re_tau),
        'scaled_residual_linf': solution.residual,
        'residual_evaluations': solution.evaluations,
        'u_center_plus': solution.center_u_plus,
        'u_bulk_plus': solution.bulk_u_plus,
        're_bulk': 2*re_tau*solution.bulk_u_plus,
        'cf_bulk': 2/solution.bulk_u_plus**2,
        'total_shear_error_linf': float(np.max(np.abs(
            solution.viscous_shear_plus+solution.reynolds_shear_plus-(1-y)))),
        'dns_re_tau': DNS_RE_TAU,
        'dns_u_center_plus': float(dns_u[-1]),
        'dns_u_bulk_plus': dns_bulk,
        'dns_u_plus_rms_difference': rms,
    }
    np.savetxt(output/'profiles.csv', np.column_stack((
        y, y*re_tau, u, solution.k_plus, solution.omega_plus, solution.nu_t_over_nu,
        solution.viscous_shear_plus, solution.reynolds_shear_plus,
    )), delimiter=',', header=(
        'wall_distance_over_h,y_plus,u_plus,k_plus,omega_plus,nu_t_over_nu,viscous_shear_plus,'
        'minus_uv_plus'
    ), comments='')
    np.savetxt(output/'comparison_vs_mkm.csv', np.column_stack((
        dns_y, dns_y_plus, comparison, dns_u, comparison-dns_u,
    )), delimiter=',', header='wall_distance_over_h,dns_y_plus,rans_u_plus,dns_u_plus,difference_u_plus',
        comments='')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 4.2), layout='constrained')
    FigureCanvasAgg(fig)
    velocity, shear = fig.subplots(1, 2)
    velocity.semilogx(y_profile[1:]*re_tau, u_profile[1:], color='.1', label='Wilcox2006-LRN')
    velocity.semilogx(dns_y_plus[1:], dns_u[1:], '--', color='.4', label='MKM DNS, Reτ = 178.12')
    velocity.set(xlabel=r'$y^+$', ylabel=r'$U^+$')
    velocity.legend(frameon=False)
    shear.plot(y, solution.viscous_shear_plus, color='.4', label='Viscous shear')
    shear.plot(y, solution.reynolds_shear_plus, color='.1', label='Reynolds shear')
    shear.plot(dns_y, 1-dns_y-dns_gradient/DNS_RE_TAU, '--', color='.45', label='MKM Reynolds shear')
    shear.plot([0, 1], [1, 0], ':', color='.2', label='Total shear')
    shear.set(xlabel=r'$y/h$', ylabel=r'Shear / $\tau_w$')
    shear.legend(frameon=False)
    for ax in (velocity, shear):
        ax.grid(alpha=.2)
    fig.savefig(output/'comparison_vs_mkm.png', dpi=175)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, default=DNS_MESH)
    parser.add_argument('--re-tau', type=float, default=DNS_RE_TAU)
    parser.add_argument('--order', type=int, default=3, help='Gauss point order for the wall-normal mesh')
    parser.add_argument('--output', type=Path, default=Path('results/channel-rans/wilcox2006_lrn'))
    args = parser.parse_args()
    mesh = CartesianMesh.read(args.mesh)
    result = solve_channel(mesh, re_tau=args.re_tau, order=args.order)
    summary = write_results(result, mesh, args.mesh, args.order, args.re_tau, args.output)
    print(f'{args.output}  residual={result.residual:.3e}  wall points={len(result.y_over_h)}  '
          f'Uc+={result.center_u_plus:.4f}  Ub+={result.bulk_u_plus:.4f}  '
          f'MKM RMS du+={summary["dns_u_plus_rms_difference"]:.4f}')
