"""FR Maxwell radiation and a closed-surface near-to-far transform.

Convention: Re(Uhat exp(-i omega t)), outgoing exp(+ikr)/r; c=eta=1.
Run from repository root: python -m case.uniform_array.solve --validate
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from fr.core import CartesianFR
from fr.equations.maxwell import Maxwell
from fr.pml import SplitPML
from fr.plotting import PlotRecorder, add_plot_arguments
from fr.time import integrate


# Phases in ascending x order, using cos(omega*t-phi_n).
PHASE_PRESETS = {
    'scan30': (0., 90., 180., 270.),
    'scan_minus30': (0., -90., -180., -270.),
    'endfire': (0., 180., 360., 540.),
    'difference': (0., 0., 180., 180.),
}


def array_factor(directions, positions, phase):
    """Complex factor, with excitation exp(+i phase) and outgoing waves."""
    return torch.exp(1j*(phase[None, :]-2*math.pi*directions[:, :1]*positions[None, :])).sum(-1)


def far_field(xyz, normals, weights, fields, directions):
    """Return r exp(-ikr) E for k=2pi, eta=1, by Love surface currents.

    fields are phasors on an outward-oriented closed surface in free space.
    Chunk directions to bound memory for refined surface quadratures.
    """
    electric, magnetic = fields[:, :3], fields[:, 3:]
    js = torch.linalg.cross(normals.to(fields.dtype), magnetic)
    ms = -torch.linalg.cross(normals.to(fields.dtype), electric)
    result = []
    for direction in directions.split(32):
        kernel = torch.exp(-2j*math.pi*(direction@xyz.T))*weights
        n, l = kernel@js, kernel@ms
        transverse = n-direction*(n*direction).sum(-1, keepdim=True)
        result.append(.5j*(transverse-torch.linalg.cross(direction.to(l.dtype), l)))
    return torch.cat(result)


class UniformArrayCase(CartesianFR):
    def __init__(self, *, elements=4, spacing=.5, steer=0., refinement=1, p=3,
                 radius=.35, pml_thickness=.5, phases=None):
        if not isinstance(elements, int) or elements < 1:
            raise ValueError('elements must be a positive integer')
        if not all(math.isfinite(v) for v in (spacing, steer, radius, pml_thickness)):
            raise ValueError('finite geometry and steering required')
        if spacing <= 0 or not -90 <= steer <= 90 or not 0 < radius < .5 or pml_thickness <= 0:
            raise ValueError('positive spacing/PML, radius in (0,.5), steer in [-90,90] required')
        if not isinstance(refinement, int) or refinement < 1:
            raise ValueError('refinement must be a positive integer')
        self.equation = Maxwell()
        self.positions = (torch.arange(elements, dtype=torch.float64)-(elements-1)/2)*spacing
        self.phase = 2*math.pi*self.positions*math.sin(math.radians(steer))
        if phases is not None:
            if steer != 0:
                raise ValueError('explicit phases and nonzero steer are mutually exclusive')
            values = torch.as_tensor(phases, dtype=torch.float64)
            if values.shape != (elements,) or not bool(torch.isfinite(values).all()):
                raise ValueError('one finite phase in degrees per element required')
            self.phase = values*math.pi/180
        self.steer, self.radius = steer if phases is None else None, radius
        # Wavelength is one; all surfaces coincide with cell faces. Leave at
        # least .5 wavelength between the source support and x PML entrance.
        sx = math.ceil(2*(float(self.positions.abs().max())+radius))*.5
        inner = ((-sx-.5, sx+.5), (-1., 1.), (-1., 1.))
        thickness = math.ceil(2*pml_thickness)*.5
        self.pml = SplitPML(self.equation, inner, thickness, reflection=1e-6)
        bounds = tuple((a-thickness, b+thickness) for a, b in inner)
        cells = tuple(round((b-a)*2*refinement) for a, b in bounds)
        super().__init__(cells, bounds, p, flux=self.pml.flux,
                         numerical_flux=self.pml.numerical_flux(self.equation.roe_flux),
                         boundary=self.pml.boundary(self.equation.pec_boundary),
                         speed=self.equation.speed, source=self.driving_source)
        self.initial = torch.zeros(self.shape+(18,), dtype=torch.float64)
        # Smooth compact spherical sources have a direction-independent radial
        # Fourier factor. Their normalized xy radiation is exactly |AF|^2.
        x, y, z = self.coords
        profiles = []
        for center in self.positions:
            r = ((x-center)**2+y*y+z*z).sqrt()
            g = torch.where(r < radius, torch.cos(math.pi*r/(2*radius))**2, 0.)
            mass = self.integral(g[..., None]).item()
            if mass <= 0:
                raise ValueError('source is unresolved; refine the mesh or increase radius')
            profiles.append(g/mass)
        self.drive_cos = sum(g*phase.cos() for g, phase in zip(profiles, self.phase))
        self.drive_sin = sum(g*phase.sin() for g, phase in zip(profiles, self.phase))
        self.damping = self.pml.rates(self.coords)[..., :, None]
        self.surface_specs = []
        xyz, normals, weights = [], [], []
        limits = ((-sx, sx), (-.5, .5), (-.5, .5))
        for axis in range(3):
            rest = [a for a in range(3) if a != axis]
            slices = [slice(round((limits[a][0]-bounds[a][0])/self.h[a]),
                            round((limits[a][1]-bounds[a][0])/self.h[a])) for a in rest]
            for side in (-1, 1):
                edge = limits[axis][int(side == 1)]
                index = round((edge-bounds[axis][0])/self.h[axis])-(side == 1)
                selection = (index, *slices)
                vector = self.L if side == -1 else self.R
                self.surface_specs.append((axis, selection, vector))
                coords = self.trace_coords[axis][int(side == 1)]
                points = torch.stack([a[selection] for a in coords], -1).reshape(-1, 3)
                normal = torch.zeros_like(points)
                normal[:, axis] = side
                shape = coords[0][selection].shape
                w = (self.w[:, None]*self.w[None, :]*self.h[rest[0]]*self.h[rest[1]]/4)
                xyz.append(points)
                normals.append(normal)
                weights.append(w.expand(shape).reshape(-1))
        self.surface_xyz, self.surface_normals, self.surface_weights = map(torch.cat, (xyz, normals, weights))

    def driving_source(self, coords, Q, t):
        result = (-self.damping*Q.reshape(*Q.shape[:-1], 3, 6)).flatten(-2)
        ramp = math.sin(math.pi*min(t/2, 1.)/2)**2
        j = ramp*(self.drive_cos*math.cos(2*math.pi*t)+self.drive_sin*math.sin(2*math.pi*t))
        # Split the impressed electric source equally among directional fields.
        for offset in (2, 8, 14):
            result[..., offset] -= j/3
        return result

    def surface(self, Q):
        U = self.pml.physical(Q)
        return torch.cat([self._trace(U.permute(self._permutation(a)), v)[s].reshape(-1, 6)
                          for a, s, v in self.surface_specs])

    def run(self, warmup=8, periods=2, cfl=.6, monitor=None):
        if not isinstance(warmup, int) or warmup < 2 or not isinstance(periods, int) or periods < 2:
            raise ValueError('warmup >= 2 and sampling periods >= 2 must be integers')
        dt = min(self.timestep(self.initial, cfl), .5/self.pml.sigma_max)
        # End each period exactly, and use trapezoidal phasor integration on
        # accepted states. Preserve individual cycles to assess settling.
        cycles = []
        U = self.initial
        stats = dict(steps=0)
        previous = None
        integral = torch.zeros((len(self.surface_xyz), 6), dtype=torch.complex128)
        def observe(Q, t):
            nonlocal previous, integral
            if monitor is not None:
                monitor(Q, t)
            if t < warmup-1e-12:
                return
            value = self.surface(Q)*complex(math.cos(2*math.pi*t), math.sin(2*math.pi*t))
            if previous is not None:
                last_t, last_value = previous
                integral += (t-last_t)*(last_value+value)  # 2/T, T=1
            previous = (t, value)
        for cycle in range(warmup+periods):
            U, step_stats = integrate(self.rhs, U, cycle+1., start_time=float(cycle), dt=dt,
                                      method='rk4', validate=self.equation.valid, monitor=observe)
            stats['steps'] += step_stats['steps']
            if cycle >= warmup:
                cycles.append(integral.clone())
                integral.zero_()
        fields = torch.stack(cycles).mean(0)
        settling = ((cycles[-1]-cycles[-2]).norm()/cycles[-1].norm().clamp(min=1e-30)).item()
        stats.update(time=warmup+periods, dt=dt, cells=self.cells, p=self.p,
                     elements=len(self.positions), spacing=float(self.positions[1]-self.positions[0]) if len(self.positions)>1 else None,
                     steer_degrees=self.steer, radius=self.radius, warmup_periods=warmup,
                     phase_degrees=(self.phase*180/math.pi).tolist(),
                     sampling_periods=periods, surface_cycle_relative_change=settling,
                     pml_thickness=self.pml.thickness)
        return U, fields, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elements', type=int, default=4)
    parser.add_argument('--spacing', type=float, default=.5, help='in wavelengths')
    parser.add_argument('--steer', type=float, default=0., help='degrees from +y toward +x; xy cut')
    excitation = parser.add_mutually_exclusive_group()
    excitation.add_argument('--preset', choices=tuple(PHASE_PRESETS), help='four-element classic phase excitation')
    excitation.add_argument('--phases', type=float, nargs='+', help='per-element phase degrees, ascending x; cos(omega*t-phase)')
    parser.add_argument('--refinement', type=int, default=1, help='cells per half wavelength')
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--radius', type=float, default=.35)
    parser.add_argument('--pml-thickness', type=float, default=.5, help='rounded up to half wavelength')
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--periods', type=int, default=2)
    parser.add_argument('--cfl', type=float, default=.6)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', help='default: results/uniform_array[/phases/PRESET]')
    parser.add_argument('--validate', action='store_true')
    add_plot_arguments(parser)
    args = parser.parse_args()
    if (args.preset or args.phases is not None) and args.steer != 0:
        parser.error('--steer cannot be combined with --preset or --phases')
    if args.preset and args.elements != 4:
        parser.error('phase presets require --elements 4')
    phases = PHASE_PRESETS[args.preset] if args.preset else args.phases
    torch.set_num_threads(args.threads)
    case = UniformArrayCase(elements=args.elements, spacing=args.spacing, steer=args.steer,
                            refinement=args.refinement, p=args.p, radius=args.radius,
                            pml_thickness=args.pml_thickness, phases=phases)
    out = Path(args.output) if args.output else Path('results/uniform_array')
    if args.output is None and args.preset:
        out = out/'phases'/args.preset
    out.mkdir(parents=True, exist_ok=True)
    plot = PlotRecorder(case, {'Ez': lambda Q: case.pml.physical(Q)[..., 2]}, args.warmup+args.periods,
                        frames=args.frames, fps=args.fps, enabled=not args.no_plot)
    U, fields, stats = case.run(args.warmup, args.periods, args.cfl, plot)
    theta = torch.linspace(-180., 180., 721, dtype=torch.float64)
    angle = theta*math.pi/180
    directions = torch.stack((angle.sin(), angle.cos(), torch.zeros_like(angle)), -1)
    E = far_field(case.surface_xyz, case.surface_normals, case.surface_weights, fields, directions)
    power = E.abs().square().sum(-1)
    if not bool(torch.isfinite(power).all()) or power.max() <= 1e-25:
        raise RuntimeError('no finite nonzero radiation measured')
    power = power/power.max()
    reference = array_factor(directions, case.positions, case.phase).abs().square()
    reference /= reference.max()
    error = ((power-reference).norm()/reference.norm()).item()
    stats.update(pattern_relative_l2=error, pattern_max_absolute_error=(power-reference).abs().max().item(),
                 peak_angle_degrees=theta[power.argmax()].item())
    stats['preset'] = args.preset
    stats['broadside_normalized_power'] = power[360].item()
    front = (theta >= -90) & (theta <= 90)
    stats['front_peak_angle_degrees'] = theta[front][power[front].argmax()].item()
    stats['validation_passed'] = error < .08 and stats['surface_cycle_relative_change'] < .03
    with (out/'pattern.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(('theta_degrees', 'fr_normalized_power', 'analytic_normalized_power'))
        writer.writerows(zip(theta.tolist(), power.tolist(), reference.tolist()))
    (out/'metrics.json').write_text(json.dumps(stats, indent=2)+'\n')
    torch.save(dict(coords=case.coords, U=case.pml.physical(U), surface_xyz=case.surface_xyz,
                    surface_normals=case.surface_normals, surface_weights=case.surface_weights,
                    surface_fields=fields, directions=directions, far_E=E, stats=stats), out/'solution.pt')
    plot.finish(out/'array', U, stats['time'])
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(angle.numpy(), power.numpy(), label='FR + surface integral')
        ax.plot(angle.numpy(), reference.numpy(), '--', label='Analytic array factor')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('Normalized power, xy plane')
        ax.legend(loc='lower right', bbox_to_anchor=(1.3, -.15))
        fig.savefig(out/'pattern.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(stats), flush=True)
    if args.validate and not stats['validation_passed']:
        raise SystemExit('validation failed: pattern L2 >= 8% or cycle change >= 3%')


if __name__ == '__main__':
    main()
