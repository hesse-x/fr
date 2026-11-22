"""Sample the jet using the element-local FR polynomial."""
import csv
import json
import math

import torch


def write_velocity_slice(model, state, output, step):
    """Save the z=0 axial-velocity section using a fixed linear color scale."""
    path = output/'slices'/f'axial_velocity_z0_step{step:06d}.png'
    if path.exists():
        return path
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.figure import Figure

    (x0, x1), (y0, y1), _ = model.s.bounds
    x, y = torch.meshgrid(torch.linspace(x0, x1, 601, dtype=model.s.z.dtype),
                          torch.linspace(y0, y1, 481, dtype=model.s.z.dtype), indexing='xy')
    axial = state[..., 1]/state[..., 0]
    values = torch.cat([
        model.probes((xs, ys, torch.zeros_like(xs)))(axial)
        for xs, ys in zip(x.flatten().split(24000), y.flatten().split(24000))
    ]).reshape(x.shape)
    norm = Normalize(vmin=-.45, vmax=1.2)
    zero = float(norm(0.))
    base = matplotlib.colormaps['RdBu_r']
    colors = [(zero*i/128, base(.5*i/128)) for i in range(129)]
    colors.extend((zero+(1-zero)*i/128, base(.5+.5*i/128)) for i in range(1, 129))
    palette = LinearSegmentedColormap.from_list('jet_velocity', colors, N=2048)
    figure = Figure(figsize=(11, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(values.numpy(), origin='lower', extent=(x0, x1, y0, y1),
                        interpolation='bilinear', cmap=palette, norm=norm, aspect='equal')
    axes.set(xlabel=r'$x/D$', ylabel=r'$y/D$',
             title=rf'Instantaneous axial velocity at $z/D=0$  |  Step {step:,}')
    figure.colorbar(image, ax=axes, ticks=[-.4, -.2, 0., .2, .4, .6, .8, 1., 1.2],
                    label=r'$u_x/U_j$')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    figure.savefig(temporary, dpi=175, format='png')
    temporary.replace(path)
    return path


class Probes:
    def __init__(self, grid, xyz):
        indices = []
        basis = []
        for edges, coord in zip(grid.edges, xyz):
            index = (torch.searchsorted(edges, coord.contiguous(), right=True)-1).clamp(0, len(edges)-2)
            local = 2*(coord-edges[index])/(edges[index+1]-edges[index])-1
            factors = torch.eye(grid.p+1, dtype=torch.bool, device=grid.z.device)
            numerator = (local[:, None, None]-grid.z).masked_fill(factors, 1.).prod(-1)
            denominator = (grid.z[:, None]-grid.z).masked_fill(factors, 1.).prod(-1)
            indices.append(index)
            basis.append(numerator/denominator)
        self.indices = tuple(indices)
        self.basis = basis

    def __call__(self, field):
        return torch.einsum('mi,mj,mk,mijk->m', *self.basis, field[self.indices])


def write_csv(path, headers, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


class FieldStatistics:
    """Online Reynolds averages and velocity covariances at all FR solution points."""

    stress_components = ('uu', 'vv', 'ww', 'uv', 'uw', 'vw')
    stress_pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def __init__(self, model):
        self.model = model
        self.moments = None

    def add(self, state, weight, total_weight):
        density, velocity, pressure = self.model.eq.primitive(state)
        values = torch.stack((density, *velocity.unbind(-1), pressure))
        if self.moments is None:
            self.moments = values, values.new_zeros((6, *density.shape))
        else:
            mean, covariance_sum = self.moments
            delta = values - mean
            fraction = weight/total_weight
            mean.add_(delta, alpha=fraction)
            velocity_delta = delta[1:4]
            for stress, (i, j) in zip(covariance_sum, self.stress_pairs):
                stress.addcmul_(velocity_delta[i], velocity_delta[j], value=weight*(1-fraction))

    def write(self, path, *, metadata, total_weight):
        mean, covariance_sum = self.moments
        stress = covariance_sum/total_weight
        grid = self.model.s
        geometry = (self.model.field_geometry() if hasattr(self.model,'field_geometry')
                    else {'edges_D': [edge.tolist() for edge in grid.edges]})
        temporary = path.with_suffix('.tmp')
        torch.save(dict(
            **metadata, **geometry, reference_nodes=grid.z.tolist(),
            velocity_components=('u', 'v', 'w'), stress_components=self.stress_components,
            density_mean_rho_inf=mean[0], velocity_mean_Uj=mean[1:4],
            pressure_mean_rho_inf_Uj2=mean[4], reynolds_stress_Uj2=stress,
            resolved_tke_Uj2=0.5*stress[:3].sum(0),
        ), temporary)
        temporary.replace(path)


class JetSamples:
    def __init__(self, model, *, full_field_statistics=False):
        self.model = model
        self.field = FieldStatistics(model) if full_field_statistics else None
        s = model.s
        self.probes = getattr(model, 'probes', lambda xyz: Probes(s, xyz))
        lx = s.bounds[0][1]
        radius = getattr(model, 'sample_radius', min(3., *(min(-lo, hi) for lo, hi in s.bounds[1:])))
        stations = [x for x in (1., 2., 5., 8., 10., 12.) if x < lx]
        self.stations = s.z.new_tensor(stations or [lx/2])
        self.x = torch.linspace(0., lx, max(161, math.ceil(10*lx)+1), dtype=s.z.dtype)
        self.r = torch.linspace(0., radius, 81, dtype=s.z.dtype)
        theta = torch.arange(32, dtype=s.z.dtype)*(2*math.pi/32)
        x, r, angle = torch.meshgrid(self.x, self.r, theta, indexing='ij')
        self.radial = self.probes((x.flatten(), (r*angle.cos()).flatten(), (r*angle.sin()).flatten()))
        self.shape = x.shape
        self.count = 0
        self.weight = 0.
        self.total = torch.zeros(x.shape[:2], dtype=s.z.dtype)

    def profile(self, state):
        return self.radial(state[..., 1]/state[..., 0]).reshape(self.shape).mean(-1)

    def add(self, state, weight=1.):
        self.count += 1
        self.weight += weight
        if self.field is None:
            profile = self.profile(state)
            self.total.add_(profile, alpha=weight)
        else:
            self.field.add(state, weight, self.weight)

    def write(self, state, output, *, time, steps, elapsed, plot, sample_from, sample_interval, cfl,
              backend, group_packet, sample_from_step=None, sample_max_steps=None,
              last_sample_time=None, sample_weighting='time'):
        if not self.count:
            mean = self.profile(state)
        elif self.field is None:
            mean = self.total/self.weight
        else:
            mean = self.radial(self.field.moments[0][1]).reshape(self.shape).mean(-1)
        center = mean[:, 0]
        half_width = []
        for row in mean:
            crossing = torch.nonzero(row[1:] <= row[0]/2)
            value = math.nan
            if row[0] > .05 and len(crossing):
                i = int(crossing[0, 0])
                fraction = (row[i]-row[0]/2)/(row[i]-row[i+1])
                value = float(torch.lerp(self.r[i], self.r[i+1], fraction))
            half_width.append(value)
        write_csv(output/'centerline.csv', ('x_D', 'u_center_Uj', 'half_width_D'),
                  zip(self.x.tolist(), center.tolist(), half_width))
        # Interpolate the sampled profiles to each requested station.
        i = torch.searchsorted(self.x, self.stations, right=True).clamp(1, len(self.x)-1)-1
        fraction = ((self.stations-self.x[i])/(self.x[i+1]-self.x[i]))[:, None]
        profiles = torch.lerp(mean[i], mean[i+1], fraction)
        write_csv(output/'radial_profiles.csv', ('x_D', 'r_D', 'u_Uj'),
                  ((float(x), float(r), float(u)) for x, profile in zip(self.stations, profiles)
                   for r, u in zip(self.r, profile)))
        metadata = dict(time_Uj_D=time, steps=steps, samples=self.count, sample_from=sample_from,
                        sample_interval=sample_interval, sample_from_step=sample_from_step)
        if sample_max_steps is not None:
            metadata.update(sample_max_steps=sample_max_steps,
                            last_sample_time=last_sample_time if self.count else None,
                            averaged_time_Uj_D=last_sample_time-sample_from if self.count else 0.)
            if sample_weighting == 'step':
                metadata['sample_weighting'] = sample_weighting
        field_file = None
        if self.field is not None and self.count:
            field_file = 'field_statistics.pt'
            self.field.write(output/field_file, metadata=metadata, total_weight=self.weight)
        (output/'summary.json').write_text(json.dumps(dict(
            **metadata, elapsed_seconds=elapsed,
            profiles='time mean' if self.count else 'instantaneous', field_statistics=field_file,
            cells=self.model.s.cells, p=self.model.s.p, sgs='WALE',
            reynolds_D=1/self.model.mu, shear_width_D=self.model.shear_width,
            mach=self.model.mach, disturbance=self.model.disturbance, cfl=cfl,
            flux=self.model.interface_flux, modal_filter=self.model.modal_filter_name,
            outlet=self.model.outlet,
            backend=backend, group_packet=group_packet, threads=torch.get_num_threads(),
        ), indent=2)+'\n')
        if plot:
            self.plot(state, output, time, center, half_width, profiles)

    def plot(self, state, output, time, center, half_width, profiles):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10, 7), layout='constrained')
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        axes[0, 0].plot(self.x, center)
        axes[0, 0].set(xlabel='x/D', ylabel='Centerline u/Uj')
        axes[0, 1].plot(self.x, half_width)
        axes[0, 1].set(xlabel='x/D', ylabel='Half-width / D')
        for x, u in zip(self.stations, profiles):
            axes[1, 0].plot(self.r, u, label=f'x/D = {float(x):g}')
        axes[1, 0].set(xlabel='r/D', ylabel='u/Uj')
        axes[1, 0].legend(fontsize='small')
        label = f'{self.count} time samples' if self.count else 'instantaneous profiles'
        figure.suptitle(f'Round jet near field, t Uj/D = {time:.3g}, {label}')

        yy = torch.linspace(-float(self.r[-1]), float(self.r[-1]), 151, dtype=self.r.dtype)
        x, y = torch.meshgrid(self.x, yy, indexing='xy')
        cut = self.probes((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()))
        u = cut(state[..., 1]/state[..., 0]).reshape(x.shape)
        image = axes[1, 1].pcolormesh(self.x, yy, u, shading='auto', cmap='viridis', vmin=0., vmax=1.)
        axes[1, 1].set(title='Instantaneous axial velocity at z/D = 0', xlabel='x/D', ylabel='y/D')
        figure.colorbar(image, ax=axes[1, 1], label='u/Uj')
        figure.savefig(output/'near_field.png', dpi=150)
