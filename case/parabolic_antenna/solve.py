"""3D FR Maxwell focal-feed parabolic reflector, wavelength=c=eta=1."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

from fr.core import CartesianFR
from fr.equations.maxwell import Maxwell
from fr.pml import SplitPML
from fr.plotting import PlotRecorder, add_plot_arguments
from fr.time import integrate
from case.uniform_array.solve import far_field


class ParabolicAntennaCase(CartesianFR):
    def __init__(self, *, diameter=3., focal_length=1., thickness=.5,
                 refinement=2, p=3, radius=.3, pml_thickness=.5, local_refinement=1,
                 feed='current', voltage_length=1/15, voltage_amplitude=1.,
                 domain_half_width=None, outer_boundary='pml'):
        if feed not in ('current', 'voltage') or outer_boundary not in ('pml', 'silver-muller'):
            raise ValueError('unknown feed or outer boundary')
        if not isinstance(local_refinement, int) or local_refinement < 1:
            raise ValueError('local_refinement must be a positive integer')
        if feed == 'voltage' and (not math.isfinite(voltage_length) or not 0 < voltage_length < diameter):
            raise ValueError('voltage edge length must be positive and smaller than diameter')
        if not math.isfinite(voltage_amplitude):
            raise ValueError('finite voltage amplitude required')
        self.feed, self.outer_boundary = feed, outer_boundary
        self.local_refinement, self.domain_half_width = local_refinement, domain_half_width
        self.voltage_amplitude, self.voltage_length = voltage_amplitude, voltage_length
        values = (diameter, focal_length, thickness, radius, pml_thickness)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('geometry parameters must be finite and positive')
        if not isinstance(refinement, int) or refinement < 1:
            raise ValueError('refinement must be a positive integer')
        if (feed == 'current' and radius >= focal_length) or thickness < .5/(refinement*local_refinement):
            raise ValueError('feed radius must be below focal length; reflector thickness must span a cell')
        self.diameter, self.focal_length, self.thickness = diameter, focal_length, thickness
        self.radius = radius
        self.equation = Maxwell()
        depth = (diameter/2)**2/(4*focal_length)
        transverse = math.ceil(2*(diameter/2+.75))*.5
        back = math.ceil(2*(thickness+.75))*.5
        front = math.ceil(2*(max(depth, focal_length+radius)+.75))*.5
        limits = ((-transverse, transverse), (-back, front), (-transverse, transverse))
        inner = tuple((a-.5, b+.5) for a, b in limits)
        if domain_half_width is not None:
            if not math.isfinite(domain_half_width) or domain_half_width <= max(diameter/2, focal_length, depth)+.25:
                raise ValueError('domain half width must enclose antenna with a sampling margin')
            inner = ((-domain_half_width, domain_half_width),)*3
            limits = tuple((a+.25, b-.25) for a,b in inner)
        pml_thickness = math.ceil(2*pml_thickness)*.5
        self.pml = SplitPML(self.equation, inner, pml_thickness, reflection=1e-6)
        bounds = tuple((a-pml_thickness, b+pml_thickness) for a, b in inner) if outer_boundary == 'pml' else inner
        cells = tuple(round((b-a)*2*refinement) for a, b in bounds)
        edges = None
        if local_refinement > 1 or feed == 'voltage':
            from .mesh import graded_edges
            coarse_h = .5/refinement
            fine_h = coarse_h/local_refinement
            local = ((-diameter/2, diameter/2),
                     (-thickness, max(depth, focal_length)+coarse_h/2),
                     (-diameter/2, diameter/2))
            edges = []
            for a in range(3):
                single = (-voltage_length/2, voltage_length/2) if feed == 'voltage' and a == 2 else None
                anchors = (*limits[a], *inner[a], *(single if single else (0.,)), focal_length if a == 1 else bounds[a][0])
                edges.append(graded_edges(bounds[a], coarse_h, fine_h, local[a], anchors, single_interval=single))
            cells = tuple(len(e)-1 for e in edges)
        super().__init__(cells, bounds, p, flux=self.pml.flux,
                         numerical_flux=self.pml.numerical_flux(self.equation.roe_flux),
                         boundary=self.pml.boundary(self.equation.pec_boundary),
                         speed=self.equation.speed, edges=edges)
        self.initial = torch.zeros(self.shape+(18,), dtype=torch.float64)
        centers = torch.meshgrid(*[(e[:-1]+e[1:])/2 for e in self.edges], indexing='ij')
        self.solid = self.inside_reflector(centers)
        if not bool(self.solid.any()):
            raise ValueError('reflector unresolved; refine the mesh')
        self.solid_nodes = self.solid[..., None, None, None]
        self.drive_cos = torch.zeros(self.shape, dtype=torch.float64)
        self.feed_ids = torch.empty(0, dtype=torch.int64)
        self.feed_coeff = self.feed_lift = torch.empty(0, dtype=torch.float64)
        if feed == 'current':
            x, y, z = self.coords
            r = (x*x+(y-focal_length)**2+z*z).sqrt()
            g = torch.where(r < radius, torch.cos(math.pi*r/(2*radius))**2, 0.)
            if bool(((g > 0) & self.solid_nodes).any()):
                raise ValueError('feed overlaps stair-step reflector; refine or reduce feed radius')
            mass = self.integral(g[..., None]).item()
            if mass <= 0:
                raise ValueError('feed unresolved; refine mesh or increase radius')
            self.drive_cos = g/mass
        else:
            from .voltage import voltage_edge
            self.feed_ids, self.feed_coeff, self.feed_lift = voltage_edge(self, voltage_length)
        self.damping = self.pml.rates(self.coords)[..., :, None]
        self.surface_specs = []
        xyz, normals, weights = [], [], []
        for axis in range(3):
            rest = [a for a in range(3) if a != axis]
            def edge_index(a, value):
                index = int(torch.searchsorted(self.edges[a], value))
                if index >= len(self.edges[a]) or abs(float(self.edges[a][index])-value) > 1e-10:
                    raise ValueError('sampling surface must coincide with mesh faces')
                return index
            slices = [slice(edge_index(a, limits[a][0]), edge_index(a, limits[a][1])) for a in rest]
            for side in (-1, 1):
                edge = limits[axis][int(side == 1)]
                index = edge_index(axis, edge)-(side == 1)
                selection = (index, *slices)
                vector = self.L if side == -1 else self.R
                self.surface_specs.append((axis, selection, vector))
                coords = self.trace_coords[axis][int(side == 1)]
                points = torch.stack([a[selection] for a in coords], -1).reshape(-1, 3)
                normal = torch.zeros_like(points)
                normal[:, axis] = side
                shape = coords[0][selection].shape
                area = self.widths[rest[0]][slices[0]][:, None]*self.widths[rest[1]][slices[1]][None, :]/4
                w = area[..., None, None]*self.w[None, None, :, None]*self.w[None, None, None, :]
                xyz.append(points)
                normals.append(normal)
                weights.append(w.expand(shape).reshape(-1))
        self.surface_xyz, self.surface_normals, self.surface_weights = map(torch.cat, (xyz, normals, weights))
        from .native import NativeMaxwell
        self.native = NativeMaxwell(self)

    def surface(self, Q):
        # Restrict to sampled cell slabs before tracing: avoid six full-volume
        # traces at every accepted sampling step.
        return torch.cat([torch.einsum('abijkv,i->abjkv',
                          self.pml.physical(Q.permute(self._permutation(a))[s]), v).reshape(-1, 6)
                          for a, s, v in self.surface_specs])

    def run(self, warmup=16, periods=2, cfl=.6, monitor=None, progress=None):
        if not isinstance(warmup, int) or warmup < 2 or not isinstance(periods, int) or periods < 2:
            raise ValueError('warmup >= 2 and sampling periods >= 2 must be integers')
        dt = self.timestep(self.initial, cfl)
        if self.outer_boundary == 'pml':
            dt = min(dt, .5/self.pml.sigma_max)
        # End each period exactly, and use trapezoidal phasor integration on
        # accepted states. Preserve individual cycles to assess settling.
        cycles = []
        U = self.initial
        stats = dict(steps=0)
        started = time.perf_counter()
        previous = None
        voltage_error = 0.
        integral = torch.zeros((len(self.surface_xyz), 6), dtype=torch.complex128)
        def observe(Q, t):
            nonlocal previous, integral, voltage_error
            if self.feed == 'voltage':
                from .voltage import voltage_signal
                voltage_error = max(voltage_error, abs(self.native.voltage(Q)-voltage_signal(t, self.voltage_amplitude)[0]))
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
                                      method=self.native.step,
                                      validate=self.equation.valid, monitor=observe)
            stats['steps'] += step_stats['steps']
            if cycle >= warmup:
                cycles.append(integral.clone())
                integral.zero_()
            if progress is not None:
                progress(cycle+1, warmup+periods, time.perf_counter()-started)
        fields = torch.stack(cycles).mean(0)
        settling = ((cycles[-1]-cycles[-2]).norm()/cycles[-1].norm().clamp(min=1e-30)).item()
        stats.update(time=warmup+periods, dt=dt, cells=self.cells, p=self.p,
                     elapsed_seconds=time.perf_counter()-started,
                     backend='native', threads=torch.get_num_threads(), bounds=self.bounds,
                     feed=self.feed, voltage_constraint_max_error=voltage_error if self.feed == 'voltage' else None,
                     voltage_length=self.voltage_length if self.feed == 'voltage' else None,
                     voltage_amplitude=self.voltage_amplitude if self.feed == 'voltage' else None,
                     outer_boundary=self.outer_boundary, domain_half_width=self.domain_half_width, local_refinement=self.local_refinement,
                     min_cell_widths=[float(w.min()) for w in self.widths],
                     max_cell_widths=[float(w.max()) for w in self.widths],
                     diameter=self.diameter, focal_length=self.focal_length, radius=self.radius,
                     reflector_thickness=self.thickness, solid_cells=int(self.solid.sum()),
                     warmup_periods=warmup, sampling_periods=periods,
                     surface_cycle_relative_change=settling, pml_thickness=self.pml.thickness if self.outer_boundary == 'pml' else 0.)
        return U, fields, stats

    def inside_reflector(self, coords):
        x, y, z = coords
        sag = (x*x+z*z)/(4*self.focal_length)
        return (x*x+z*z <= (self.diameter/2)**2) & (y <= sag) & (y >= sag-self.thickness)

    def rhs(self, U, t=0.):
        return self.native.rhs(U, t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (('diameter', 3.), ('focal-length', 1.), ('thickness', .5),
                          ('radius', .3), ('pml-thickness', .5), ('cfl', .6)):
        parser.add_argument('--'+name, type=float, default=default)
    for name, default in (('refinement', 2), ('p', 3), ('warmup', 16), ('periods', 2), ('threads', 1)):
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--output', default='results/parabolic_antenna')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--local-refinement', type=int, default=1)
    parser.add_argument('--feed', choices=('current', 'voltage'), default='current')
    parser.add_argument('--voltage-length', type=float, default=1/15)
    parser.add_argument('--voltage-amplitude', type=float, default=1.)
    parser.add_argument('--domain-half-width', type=float)
    parser.add_argument('--outer-boundary', choices=('pml', 'silver-muller'), default='pml')
    add_plot_arguments(parser)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    case = ParabolicAntennaCase(**{k: getattr(args, k) for k in
        ('diameter', 'focal_length', 'thickness', 'radius', 'pml_thickness', 'refinement', 'p', 'local_refinement', 'feed', 'voltage_length', 'voltage_amplitude', 'domain_half_width', 'outer_boundary')})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plot = PlotRecorder(case, {'Ez': lambda Q: case.pml.physical(Q)[..., 2]},
                        args.warmup+args.periods, frames=args.frames, fps=args.fps, enabled=not args.no_plot)
    U, fields, stats = case.run(args.warmup, args.periods, args.cfl, plot,
        progress=lambda cycle, total, elapsed: print(
            f'cycle {cycle}/{total}, elapsed {elapsed:.1f}s', file=sys.stderr, flush=True))
    theta = torch.linspace(-180., 180., 721, dtype=torch.float64)
    angle = theta*math.pi/180
    directions = torch.stack((angle.sin(), angle.cos(), torch.zeros_like(angle)), -1)
    E = far_field(case.surface_xyz, case.surface_normals, case.surface_weights, fields, directions)
    power = E.abs().square().sum(-1)
    if not bool(torch.isfinite(power).all()) or power.max() <= 1e-25:
        raise RuntimeError('no finite nonzero radiation measured')
    power /= power.max()
    stats.update(peak_angle_degrees=theta[power.argmax()].item(),
                 forward_backward_ratio_db=10*math.log10(max(power[360].item(), 1e-30)/max(power[0].item(), 1e-30)))
    stats['validation_passed'] = stats['surface_cycle_relative_change'] < .03
    with (out/'pattern.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(('theta_degrees', 'fr_normalized_power'))
        writer.writerows(zip(theta.tolist(), power.tolist()))
    (out/'metrics.json').write_text(json.dumps(stats, indent=2)+'\n')
    torch.save(dict(coords=case.coords, edges=case.edges, U=case.pml.physical(U), solid=case.solid,
                    surface_xyz=case.surface_xyz, surface_normals=case.surface_normals,
                    surface_weights=case.surface_weights, surface_fields=fields,
                    directions=directions, far_E=E, stats=stats), out/'solution.pt')
    plot.finish(out/'reflector', U, stats['time'])
    if not args.no_plot:
        from .visualize import field_plot
        field_plot(case, case.pml.physical(U), out, stats['time'])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(angle.numpy(), power.numpy(), label='FR + closed surface integral')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('Parabolic reflector: xy power cut')
        ax.legend(loc='lower right')
        fig.savefig(out/'pattern.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(stats), flush=True)
    if args.validate and not stats['validation_passed']:
        raise SystemExit('validation failed: surface cycle change >= 3%')


if __name__ == '__main__':
    main()
