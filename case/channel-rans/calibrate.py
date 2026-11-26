"""Fit the Wilcox low-Reynolds-number damping to MKM channel profiles."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from fr import CartesianMesh
from .solve import DNS_MESH, DNS_REFERENCE, DNS_RE_TAU, STANDARD_DAMPING, WilcoxDamping, solve_channel


OUTPUT = Path('results/channel-rans/calibration')
HOLDOUT_RE_TAU = 392.24
PROFILE_NAMES = ('u_plus', 'minus_uv_plus', 'k_plus')


def read_reference(means_path, stress_path, re_tau):
    mean_y, u = np.loadtxt(means_path, usecols=(0, 2)).T
    stress_y, uu, vv, ww, uv = np.loadtxt(stress_path, usecols=(0, 2, 3, 4, 5)).T
    y_plus = np.expm1(np.linspace(0., np.log1p(re_tau), 256))
    y = y_plus/re_tau
    profiles = np.array([
        PchipInterpolator(mean_y, u)(y),
        PchipInterpolator(stress_y, -uv)(y),
        PchipInterpolator(stress_y, (uu+vv+ww)/2)(y),
    ])
    return y, profiles


def evaluate(solution, y, re_tau):
    nodes = solution.y_over_h
    shear = PchipInterpolator(np.r_[0., nodes, 1.], np.r_[1., solution.viscous_shear_plus, 0.])
    u = re_tau*shear.antiderivative()(y)
    # Reflect the interior values to impose symmetry when sampling k near the center.
    k = PchipInterpolator(np.r_[0., nodes, 2.-nodes[::-1]],
                          np.r_[0., solution.k_plus, solution.k_plus[::-1]])(y)
    return np.array([u, 1.-y-shear(y), k])


def solve_perturbed(mesh, re_tau, log_ratios, standard_solution):
    standard = np.array(list(asdict(STANDARD_DAMPING).values()))
    steps = max(1, int(np.ceil(np.max(np.abs(log_ratios))/.20)))
    solution = standard_solution
    for fraction in np.linspace(0., 1., steps+1)[1:]:
        damping = WilcoxDamping(*(standard*np.exp(fraction*log_ratios)))
        solution = solve_channel(mesh, re_tau=re_tau, damping=damping, initial_solution=solution)
    return solution


def bisect_wall_normal(mesh):
    x, y, z = (axis.numpy() for axis in mesh.edges)
    y_refined = np.empty(2*len(y)-1)
    y_refined[::2] = y
    y_refined[1::2] = (y[:-1]+y[1:])/2
    return CartesianMesh((x, y_refined, z))


def compare(mesh, re_tau, y, dns, log_ratios):
    standard = solve_channel(mesh, re_tau=re_tau)
    fitted = solve_perturbed(mesh, re_tau, log_ratios, standard)
    old, new = (evaluate(solution, y, re_tau) for solution in (standard, fitted))
    metrics = {
        're_tau': re_tau, 'gauss_order': 3, 'wall_normal_cells': mesh.cells[1],
        'first_y_plus': standard.y_over_h[0]*re_tau,
    }
    for label, predicted, solution in (('standard', old, standard), ('fitted', new, fitted)):
        metrics[label] = {
            'rmse': dict(zip(PROFILE_NAMES, np.sqrt(np.mean((predicted-dns)**2, axis=1)))),
            'u_center_plus': solution.center_u_plus,
            'u_bulk_plus': solution.bulk_u_plus,
            'u_plus_at_y_plus_25_342': np.interp(25.342, re_tau*y, predicted[0]),
            'minus_uv_plus_at_y_plus_15': np.interp(15., re_tau*y, predicted[1]),
            'scaled_residual_linf': solution.residual,
        }
    metrics['dns'] = {
        'u_center_plus': dns[0, -1],
        'u_bulk_plus': float(PchipInterpolator(y, dns[0]).integrate(0., 1.)),
        'u_plus_at_y_plus_25_342': np.interp(25.342, re_tau*y, dns[0]),
        'minus_uv_plus_at_y_plus_15': np.interp(15., re_tau*y, dns[1]),
    }
    return metrics, old, new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=OUTPUT/'reference')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--include-k', action='store_true',
                        help='include the turbulent kinetic energy profile')
    args = parser.parse_args()
    reference = args.reference_dir
    output = args.output or (OUTPUT/'with_k' if args.include_k else OUTPUT)

    mesh = CartesianMesh.read(DNS_MESH)
    train = read_reference(DNS_REFERENCE, reference/'chan180.reystress', DNS_RE_TAU)
    holdout = read_reference(reference/'chan395.means', reference/'chan395.reystress', HOLDOUT_RE_TAU)
    baseline = solve_channel(mesh)
    baseline_error = evaluate(baseline, train[0], DNS_RE_TAU)-train[1]
    quantity_count = 3 if args.include_k else 2
    scales = np.sqrt(np.mean(baseline_error[:quantity_count]**2, axis=1))

    def objective(log_ratios):
        sol = solve_perturbed(mesh, DNS_RE_TAU, log_ratios, baseline)
        error = evaluate(sol, train[0], DNS_RE_TAU)-train[1]
        return (error[:quantity_count]/scales[:, None]).ravel()

    fit = least_squares(objective, np.zeros(3), bounds=(-np.log(4.), np.log(4.)),
                        jac='3-point', diff_step=1e-3, ftol=1e-10, xtol=1e-10, gtol=1e-9)
    if not fit.success:
        raise RuntimeError(f'Damping fit failed: {fit.message}')
    coefficients = dict(zip(asdict(STANDARD_DAMPING),
                            np.array(list(asdict(STANDARD_DAMPING).values()))*np.exp(fit.x)))
    summary = {
        'standard_damping': asdict(STANDARD_DAMPING), 'fitted_damping': coefficients,
        'fit_profiles': PROFILE_NAMES[:quantity_count],
        'parameter_ratio_bounds': [0.25, 4.0],
        'active_parameter_bounds': fit.active_mask.tolist(),
        'optimizer_optimality': float(fit.optimality),
        'fit_objective_relative_to_standard': float(np.mean(fit.fun**2)),
    }
    mesh_refined = bisect_wall_normal(mesh)
    output.mkdir(parents=True, exist_ok=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 6.4), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, quantity_count)
    for row, (label, grid, re_tau, data) in enumerate((
        ('fit', mesh, DNS_RE_TAU, train), ('holdout', mesh_refined, HOLDOUT_RE_TAU, holdout),
    )):
        y, dns = data
        summary[label], old, new = compare(grid, re_tau, y, dns, fit.x)
        np.savetxt(output/f'{label}_profiles.csv', np.column_stack((y*re_tau, dns.T, old.T, new.T)),
                   delimiter=',', comments='', header='y_plus,'+','.join(
                       f'{source}_{name}' for source in ('dns', 'standard', 'fitted') for name in PROFILE_NAMES))
        labels = (r'$U^+$', r'$-\overline{u\prime v\prime}^+$', r'$k^+$')
        for quantity, (ax, ylabel) in enumerate(zip(axes[row], labels)):
            ax.semilogx(y[1:]*re_tau, dns[quantity, 1:], 'k-', label='MKM DNS')
            ax.semilogx(y[1:]*re_tau, old[quantity, 1:], '--', color='.45', label='Standard coefficients')
            ax.semilogx(y[1:]*re_tau, new[quantity, 1:], color='#146b82', label='Fitted coefficients')
            role = 'Fit' if label == 'fit' else 'Holdout'
            ax.set(xlabel=r'$y^+$', ylabel=ylabel, title=f'{role}, '+rf'$Re_\tau={re_tau:.2f}$',
                   xlim=(.5, re_tau))
            ax.grid(alpha=.2)
    summary['fit_refined'] = compare(mesh_refined, DNS_RE_TAU, *train, fit.x)[0]
    summary['holdout_refined'] = compare(bisect_wall_normal(mesh_refined), HOLDOUT_RE_TAU, *holdout, fit.x)[0]
    axes[0, 0].legend(frameon=False)
    fig.savefig(output/'comparison.png', dpi=175)
    report = json.dumps(summary, indent=2)
    (output/'summary.json').write_text(report+'\n')
    print(report)


if __name__ == '__main__':
    main()
