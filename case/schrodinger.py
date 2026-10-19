"""Five time-dependent Schrodinger benchmarks, hbar=m=1.

python -m case.schrodinger --case gaussian --convergence --no-plot
python -m case.schrodinger --case double-slit
"""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from fr import CartesianFR, CartesianLDG, central, integrate
from fr.equations.schrodinger import Schrodinger
from fr.plotting import PlotRecorder, add_plot_arguments


NAMES = ('gaussian', 'harmonic', 'well', 'barrier', 'double-slit')
DEFAULT_TIME = dict(gaussian=2., harmonic=2*math.pi, well=4/(3*math.pi), barrier=4., **{'double-slit': 3.})


def state(psi):
    return torch.stack((psi.real, psi.imag), -1)


def gaussian(x, t, *, x0=-4., k=2., sigma=1.):
    z = 1+1j*t/(2*sigma*sigma)
    return (2*math.pi*sigma*sigma)**(-.25)/z**.5*torch.exp(
        -(x-x0-k*t)**2/(4*sigma*sigma*z)+1j*k*(x-x0)-.5j*k*k*t)


class SchrodingerCase:
    def __init__(self, name='gaussian', n=None, p=3):
        if name not in NAMES:
            raise ValueError(f'choose one of {NAMES}')
        self.name = name
        self.final_time = DEFAULT_TIME[name]
        dim = 2 if name == 'double-slit' else 1
        n = (32 if dim == 2 else (16 if name == 'well' else 48)) if n is None else n
        bounds = ((-12., 12.), (-8., 8.)) if dim == 2 else (((0., 1.),) if name == 'well' else ((-16., 16.),))
        # FR grid only supplies geometry and reference operators here.
        self.grid = CartesianFR((n,)*dim, bounds, p, flux=lambda U,a: U,
                                numerical_flux=central, periodic=False,
                                boundary=lambda c,U,a,s,t: -U)
        self.ldg = CartesianLDG(self.grid, boundary=lambda c,U,a,s,t: torch.zeros_like(U))
        self.equation = Schrodinger(self.ldg, self.potential)
        self.initial = self.exact(0.) if name in ('gaussian','harmonic','well') else self.packet()

    def potential(self, coords, t):
        x = coords[0]
        if self.name == 'harmonic':
            return .5*x*x
        if self.name == 'barrier':
            # Faces align with both edges for default n=48.
            return 6.*((x >= 0.) & (x < 2/3)).to(x.dtype)
        if self.name == 'double-slit':
            y = coords[1]
            # Finite-height screen, two apertures centered at y=+-1.5.
            openings = ((y-1.5).abs() < .5) | ((y+1.5).abs() < .5)
            return 30.*((x >= 0.) & (x < .75) & ~openings).to(x.dtype)
        return torch.zeros_like(x)

    def exact(self, t):
        x = self.grid.coords[0]
        if self.name == 'gaussian':
            psi = gaussian(x, t)
        elif self.name == 'harmonic':
            # Displaced oscillator ground state, x0=2, p0=0.
            q, momentum = 2*math.cos(t), -2*math.sin(t)
            psi = math.pi**(-.25)*torch.exp(-.5*(x-q)**2+1j*(momentum*(x-q/2)-t/2))
        elif self.name == 'well':
            psi = torch.sin(math.pi*x)*complex(math.cos(-.5*math.pi**2*t), math.sin(-.5*math.pi**2*t))
            psi = psi+torch.sin(2*math.pi*x)*complex(math.cos(-2*math.pi**2*t), math.sin(-2*math.pi**2*t))
        else:
            raise ValueError('scattering cases have no implemented analytic time-dependent reference')
        return state(psi)

    def packet(self):
        x = self.grid.coords[0]
        if self.name == 'barrier':
            return state(gaussian(x, 0., x0=-6., k=3., sigma=.8))
        y = self.grid.coords[1]
        return state(gaussian(x, 0., x0=-6., k=4., sigma=.8)*gaussian(y, 0., x0=0., k=0., sigma=2.))

    def timestep(self):
        # Conservative explicit RK4 bound: LDG spectral scale ~ (p+1)^4/h^2.
        rate = (self.grid.p+1)**4*sum(h**-2 for h in self.grid.h)
        rate += float(self.potential(self.grid.coords, 0.).abs().max())
        return .5/rate

    def run(self, final_time=None, *, frames=60, fps=12, output=None, plot=True):
        final_time = self.final_time if final_time is None else final_time
        s, eq = self.grid, self.equation
        recorder = PlotRecorder(s, {'Probability density': lambda U: U.square().sum(-1),
                                   'Real psi': lambda U: U[...,0]}, final_time,
                                frames=frames, fps=fps, enabled=plot)
        history = []
        target = 0.
        def monitor(U,t):
            nonlocal target
            recorder(U,t)
            if t+1e-12 >= target or t == final_time:
                history.append((t,float(eq.probability(U)),float(eq.energy(U))))
                target = t+final_time/100 if final_time else 1.
        U, stats = integrate(eq.rhs, self.initial, final_time, dt=self.timestep(), method='rk4', monitor=monitor)
        mass0 = float(eq.probability(self.initial))
        energy0 = float(eq.energy(self.initial))
        density = U.square().sum(-1, keepdim=True)
        stats.update(case=self.name, n=s.cells[0], p=s.p, dt_limit=self.timestep(),
                     probability_initial=mass0, probability_final=float(eq.probability(U)),
                     probability_change=float(eq.probability(U))-mass0,
                     energy_initial=energy0, energy_change=float(eq.energy(U))-energy0)
        if self.name in ('gaussian','harmonic','well'):
            error = U-self.exact(final_time)
            stats['complex_l2_error'] = float(s.integral(error.square().sum(-1,keepdim=True))[0].sqrt())
        if self.name in ('barrier','double-slit'):
            x = s.coords[0]
            edge = 2/3 if self.name == 'barrier' else .75
            stats['left_probability'] = float(s.integral(density*(x<0)[...,None])[0])/mass0
            stats['screen_probability'] = float(s.integral(density*((x>=0)&(x<edge))[...,None])[0])/mass0
            stats['right_probability'] = float(s.integral(density*(x>=edge)[...,None])[0])/mass0
            stats['partition_note'] = 'finite-time region probabilities; not asymptotic reflection/transmission coefficients'
        if output is not None:
            folder = Path(output)
            folder.mkdir(parents=True, exist_ok=True)
            recorder.finish(folder/self.name, U, stats['time'])
            (folder/'metrics.json').write_text(json.dumps(stats, indent=2)+'\n')
            torch.save(dict(coords=s.coords, initial=self.initial, solution=U,
                            potential=self.potential(s.coords,stats['time']), time=stats['time']), folder/'solution.pt')
            with (folder/'history.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(('time','probability','energy')); writer.writerows(history)
            with (folder/'field.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow([f'x{a}' for a in range(s.dim)]+['real','imag','density','potential'])
                data = torch.stack((*s.coords,U[...,0],U[...,1],density[...,0],self.potential(s.coords,stats['time'])),-1).reshape(-1,s.dim+4)
                writer.writerows(data.tolist())
        return U, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=(*NAMES,'all'), default='gaussian')
    parser.add_argument('--n', type=int)
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--time', type=float)
    parser.add_argument('--output', default='results/schrodinger')
    parser.add_argument('--convergence', action='store_true', help='three grids; analytic cases only')
    add_plot_arguments(parser)
    args = parser.parse_args()
    if args.convergence and args.case not in ('gaussian','harmonic','well'):
        parser.error('--convergence requires gaussian, harmonic or well')
    torch.set_num_threads(1)
    reports = []
    for name in NAMES if args.case == 'all' else (args.case,):
        sizes = [args.n or 12]
        if args.convergence:
            sizes += [2*sizes[0],4*sizes[0]]
        for n in sizes if args.convergence else [args.n]:
            case = SchrodingerCase(name,n,args.p)
            output = Path(args.output)/name
            if args.convergence: output = output/f'n{n}'
            _, stats = case.run(args.time, frames=args.frames, fps=args.fps, output=output, plot=not args.no_plot)
            if args.convergence and reports:
                stats['observed_order'] = math.log2(reports[-1]['complex_l2_error']/stats['complex_l2_error'])
            reports.append(stats)
            print(json.dumps(stats,indent=2),flush=True)
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    (root/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__':
    main()
