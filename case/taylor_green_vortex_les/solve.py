"""Taylor–Green LES; baseline uses Vreman, Rusanov faces and no dealiasing."""
import argparse
import math
from pathlib import Path
import torch
from fr.les import CartesianLESMixin
from case.taylor_green_vortex.solve import TaylorGreenVortex, run as run_tgv


class TaylorGreenVortexLES(CartesianLESMixin, TaylorGreenVortex):
    def __init__(self, cells=(32, 32, 32), p=3, **kwargs):
        super().__init__(cells, p, **kwargs)

    def diagnostics(self, U, t=0., *, rhs=None):
        result = super().diagnostics(U, t, rhs=rhs)
        if self.sgs is None:
            result.update(epsilon_sgs=0., mu_sgs_mean=0., mu_sgs_max=0., mu_sgs_over_mu_max=0.)
            return result
        gradient = self.viscous_operator.gradient(self.primitive(U), t)
        mu_t = self.sgs.viscosity(U[..., 0], gradient, self.delta)
        du = gradient[..., :3, :]
        strain = .5*(du+du.transpose(-1, -2))
        divergence = du.diagonal(dim1=-2, dim2=-1).sum(-1)
        def avg(v):
            return self.s.integral(v[..., None])[0]/self.volume
        epsilon_sgs = float(avg(2*mu_t*(strain.square().sum((-1, -2))-divergence.square()/3)))
        result.update(epsilon_sgs=epsilon_sgs, mu_sgs_mean=float(avg(mu_t)),
                      mu_sgs_max=float(mu_t.max()), mu_sgs_over_mu_max=float(mu_t.max()/self.mu))
        result['kinetic_budget_residual'] -= epsilon_sgs
        return result


def run(output='results/taylor_green_vortex_les', *, sgs_model='vreman', cv=.07, cs=.1, filter_ratio=1.,
        turbulent_prandtl=.9, interface_flux='rusanov', anti_alias='none',
        quadrature_order=None, filter_width='dof', **kwargs):
    """Run the original convection baseline unless framework modes are selected."""
    kwargs.setdefault('cells', (32,)*3)
    kwargs.setdefault('backend', 'native-p3')
    kwargs.setdefault('cfl', .3)
    model_options = dict(sgs_model=sgs_model, cv=cv, turbulent_prandtl=turbulent_prandtl)
    if sgs_model == 'smagorinsky':
        model_options['cs'] = cs
    if filter_ratio != 1.:
        model_options['filter_ratio'] = filter_ratio
    if filter_width != 'dof':
        model_options['filter_width'] = filter_width
    return run_tgv(output, model_class=TaylorGreenVortexLES,
                   interface_flux=interface_flux, anti_alias=anti_alias, quadrature_order=quadrature_order,
                   model_options=model_options, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, help='Cartesian JSON mesh; overrides cell counts')
    parser.add_argument('--n', type=int)
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--sgs', dest='sgs_model', choices=('none', 'vreman', 'smagorinsky'), default='vreman')
    parser.add_argument('--cv', type=float, default=.07)
    parser.add_argument('--cs', type=float, default=.1)
    parser.add_argument('--filter-ratio', type=float, default=1., help='multiplier of selected SGS width')
    parser.add_argument('--filter-width', choices=('dof', 'gauss-center', 'half-dof'), default='dof')
    parser.add_argument('--interface-flux', choices=('rusanov', 'roe'), default='rusanov',
                        help='public framework face flux; baseline: rusanov')
    parser.add_argument('--backend', choices=('native-p3', 'torch'), default='native-p3')
    parser.add_argument('--turbulent-prandtl', type=float, default=.9)
    parser.add_argument('--reynolds', type=float, default=1600.)
    parser.add_argument('--mach', type=float, default=.3)
    parser.add_argument('--time', type=float, default=10.)
    parser.add_argument('--cfl', type=float, default=.3)
    parser.add_argument('--max-dt', type=float)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--anti-alias', default='none',
                        help='public framework overintegration mode; baseline: none')
    parser.add_argument('--quadrature-order', type=int)
    parser.add_argument('--diagnostic-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    parser.add_argument('--compare-with', type=Path, nargs='*', default=[])
    parser.add_argument('--snapshot-times', type=float, nargs='*', default=[10.])
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--no-fields', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/taylor_green_vortex_les'))
    args = vars(parser.parse_args())
    if args['threads'] < 1:
        parser.error('positive threads required')
    torch.set_num_threads(args.pop('threads'))
    if args['device'] == 'cpu':
        from fr.native.cartesian_ns.runtime import configure_cpu_memory_reuse
        configure_cpu_memory_reuse()
    n = args.pop('n')
    if args['mesh'] is None and n is None:
        args['mesh'] = Path(__file__).with_name('mesh')/'default.json'
    args['cells'] = (32 if n is None else n,)*3
    args['final_time'] = args.pop('time')
    args['max_steps'] = args.pop('steps')
    args['plot'] = not args.pop('no_plot')
    args['fields'] = not args.pop('no_fields')
    run(**args)
