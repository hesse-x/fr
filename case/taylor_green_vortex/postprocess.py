"""Periodic uniform sampling, spectra, VTK snapshots and reference comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from fr.plotting import _lagrange

REFERENCE = Path(__file__).with_name('reference')/'flexi-ma01-n7-64.csv'


def uniform_sample(model, value, points=None):
    """Interpolate each cell to equispaced centers, excluding both cell faces.

    Cell-major/nodal-major input -> globally ordered (Nx,Ny,Nz,components).
    Uniform box required. Gauss nodes themselves are not FFT grid points.
    """
    m = model.s.p+1 if points is None else points
    if not isinstance(m, int) or m < 1:
        raise ValueError('positive integer sample points per cell required')
    for widths in model.s.widths:
        if not torch.allclose(widths, widths[0].expand_as(widths)):
            raise ValueError('uniform sampling requires uniform cell widths')
    target = (torch.arange(m, dtype=model.s.z.dtype, device=model.s.z.device)+.5)*2/m-1
    B = _lagrange(model.s.z, target)
    out = torch.einsum('ia,xyzabcq->xyzibcq', B, value)
    out = torch.einsum('jb,xyzibcq->xyzijcq', B, out)
    out = torch.einsum('kc,xyzijcq->xyzijkq', B, out)
    return out.permute(0, 3, 1, 4, 2, 5, 6).reshape(*(n*m for n in model.s.cells), value.shape[-1])


def spectrum(velocity, density):
    """Full-complex FFT, normalization='forward', shells floor(|k|+1/2).

    Both spectra include the mean mode. All cubic Fourier modes are retained
    for a Parseval check; bins outside the inscribed Nyquist sphere are marked.
    ``sqrt(rho)*u`` gives the sampled density-weighted kinetic energy spectrum.
    """
    if velocity.ndim != 4 or velocity.shape[-1] != 3 or density.shape != velocity.shape[:-1]:
        raise ValueError('expected uniform velocity (Nx,Ny,Nz,3) and density (Nx,Ny,Nz)')
    if not bool(torch.isfinite(velocity).all() and torch.isfinite(density).all() and (density > 0).all()):
        raise ValueError('finite velocity and positive density required')
    shape = velocity.shape[:3]
    modes = [torch.fft.fftfreq(n, d=1/n, device=velocity.device, dtype=velocity.dtype) for n in shape]
    k2 = modes[0][:, None, None].square()+modes[1][None, :, None].square()+modes[2][None, None, :].square()
    bins = (k2.sqrt()+.5).floor().to(torch.int64).reshape(-1)
    count = torch.bincount(bins).cpu().numpy()
    energies = []
    for field in (velocity, density.sqrt()[..., None]*velocity):
        fourier = torch.fft.fftn(field, dim=(0, 1, 2), norm='forward')
        energy = .5*fourier.abs().square().sum(-1)
        energies.append(torch.bincount(bins, weights=energy.reshape(-1), minlength=len(count)).cpu().numpy())
    rows = [dict(wavenumber=k, velocity_energy=float(energies[0][k]),
                 density_weighted_energy=float(energies[1][k]), mode_count=int(count[k]),
                 inside_nyquist_sphere=bool(k+.5 <= min(shape)/2)) for k in range(len(count))]
    return rows


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_vtk(path, model, fields):
    """Binary legacy structured points, centered samples and x-fast ordering."""
    shape = next(iter(fields.values())).shape[:3]
    spacing = [(hi-lo)/n for (lo, hi), n in zip(model.s.bounds, shape)]
    origin = [lo+h/2 for (lo, _), h in zip(model.s.bounds, spacing)]
    with Path(path).open('wb') as stream:
        def line(s):
            stream.write((s+'\n').encode())
        line('# vtk DataFile Version 3.0')
        line('Taylor-Green vortex; uniformly interpolated cell-center samples')
        line('BINARY')
        line('DATASET STRUCTURED_POINTS')
        line('DIMENSIONS '+' '.join(map(str, shape)))
        line('ORIGIN '+' '.join(map(str, origin)))
        line('SPACING '+' '.join(map(str, spacing)))
        line(f'POINT_DATA {math.prod(shape)}')
        for name, value in fields.items():
            if value.shape[-1] == 3:
                line(f'VECTORS {name} double')
            elif value.shape[-1] == 1:
                line(f'SCALARS {name} double 1')
                line('LOOKUP_TABLE default')
            else:
                raise ValueError('VTK fields must be scalar or vector')
            array = value.permute(2, 1, 0, 3).contiguous().cpu().numpy().astype('>f8')
            stream.write(array.tobytes())
            line('')


@torch.no_grad()
def export_snapshot(model, U, t, prefix, *, sample_points=None, fields=True, plot=True):
    prefix = Path(prefix)
    rho, velocity, pressure, gradient, omega, q = model.flow_fields(U)
    # Free the 12-component gradient before uniform interpolation / FFT.
    del gradient
    samples = dict(density=uniform_sample(model, rho[..., None], sample_points),
                   pressure=uniform_sample(model, pressure[..., None], sample_points),
                   velocity=uniform_sample(model, velocity, sample_points),
                   vorticity=uniform_sample(model, omega, sample_points),
                   Q=uniform_sample(model, q[..., None], sample_points))
    rows = spectrum(samples['velocity'], samples['density'][..., 0])
    write_csv(prefix.with_name(prefix.name+'.spectrum.csv'), rows)
    metadata = dict(time=t, uniform_shape=list(samples['density'].shape[:3]),
                    interpolation='cellwise tensor Lagrange at equispaced subcell centers',
                    fft='full complex, norm=forward; all cubic modes, no mean subtraction',
                    shells='floor(norm(integer_wavevector)+0.5); k=0 included',
                    velocity_spectral_sum=sum(r['velocity_energy'] for r in rows),
                    density_weighted_spectral_sum=sum(r['density_weighted_energy'] for r in rows),
                    sampled_kinetic_energy=float((.5*samples['density'][..., 0]*samples['velocity'].square().sum(-1)).mean()))
    prefix.with_name(prefix.name+'.sampling.json').write_text(json.dumps(metadata, indent=2)+'\n')
    if fields:
        write_vtk(prefix.with_name(prefix.name+'.vtk'), model, samples)
    if plot:
        import os
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/fr-matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        vort = samples['vorticity'].square().sum(-1).sqrt()
        image = axes[0].imshow(vort[:, :, vort.shape[2]//2].T.cpu(), origin='lower',
                               extent=(-math.pi, math.pi, -math.pi, math.pi), cmap='magma')
        axes[0].set(xlabel='x', ylabel='y', title=f'|vorticity|, center z sample; t={t:g}')
        fig.colorbar(image, ax=axes[0])
        axes[1].loglog([r['wavenumber'] for r in rows[1:]], [r['velocity_energy'] for r in rows[1:]])
        axes[1].set(xlabel='Shell k', ylabel='Velocity E(k)', title='Uniform-sample spectrum')
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name+'.png'), dpi=160)
        plt.close(fig)


def load_reference(path=REFERENCE):
    path = Path(path)
    if path == REFERENCE:
        provenance = json.loads(path.with_name('provenance.json').read_text())
        expected = next(f['sha256'] for f in provenance['files'] if f['path'] == path.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('reference checksum mismatch')
    with path.open() as stream:
        rows = [{k:float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if len(rows) < 3 or any(b['Time'] <= a['Time'] for a, b in zip(rows, rows[1:])):
        raise ValueError('reference times must be strictly increasing')
    return rows


def compare_runs(directories, output, *, reference=REFERENCE, plot=True):
    """Compare identical definitions; kinetic RHS decay is a separate diagnostic.

    Reference 'Dissipation Rate Compressible' is DR_Sd - <p div u>,
    not the time derivative of sampled kinetic energy. Match that definition.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ref = load_reference(reference)
    rt = np.array([r['Time'] for r in ref])
    mapping = dict(kinetic_energy='Ekin comp', kinetic_energy_unweighted='Ekin incomp',
                   epsilon_strain='DR_S', epsilon_viscous='DR_Sd',
                   epsilon_velocity_gradient='DR_u', epsilon_vorticity='ED_S',
                   epsilon_dilatation='ED_D', enstrophy='Enstrophy comp',
                   max_vorticity='Maximum Vorticity', uprime='uprime')
    reference_curves = {name:np.array([r[col] for r in ref]) for name, col in mapping.items()}
    reference_curves['epsilon_physical'] = np.array([r['Dissipation Rate Compressible'] for r in ref])
    reference_curves['pressure_dilatation'] = -np.array([r['DR_p'] for r in ref])
    reference_curves['mean_temperature_normalized'] = np.array([r['Mean Temperature'] for r in ref])/ref[0]['Mean Temperature']
    reference_curves['entropy_change'] = np.array([r['Mean Entropy'] for r in ref])-ref[0]['Mean Entropy']
    reference_curves['epsilon_kinetic'] = -np.gradient(reference_curves['kinetic_energy'], rt, edge_order=2)
    results, curves = [], []
    for directory in directories:
        directory = Path(directory)
        config = json.loads((directory/'run.json').read_text())['config']
        if config['reynolds'] != 1600. or config['gamma'] != 1.4 or config['prandtl'] != .71:
            raise ValueError('reference requires Re=1600, gamma=1.4, Pr=.71')
        with (directory/'history.csv').open() as stream:
            history = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        times = np.array([r['time'] for r in history])
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError('at least two strictly increasing diagnostic times required')
        if times[0] < rt[0] or times[-1] > rt[-1]:
            raise ValueError('comparison would extrapolate beyond reference times')
        metrics, differences, plot_data = {}, [], {}
        own_curves = {name:np.array([r[name] for r in history]) for name in reference_curves
                      if all(name in r for r in history)}
        own_curves['epsilon_physical'] = np.array([r['epsilon_viscous']-r['pressure_dilatation'] for r in history])
        if all('mean_entropy' in r for r in history):
            own_curves['entropy_change'] = np.array([r['mean_entropy'] for r in history])-history[0]['mean_entropy']
        for name, ours in own_curves.items():
            values = reference_curves[name]
            expected = np.interp(times, rt, values)
            error = ours-expected
            trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
            duration = times[-1]-times[0]
            ref_norm = trap(expected**2, times)
            peak_mask = (rt >= times[0]) & (rt <= times[-1])
            ref_peak_indices = np.flatnonzero(peak_mask)
            signed = name in ('pressure_dilatation', 'entropy_change')
            reference_peak = ref_peak_indices[np.argmax(np.abs(values[peak_mask]) if signed else values[peak_mask])]
            own_peak = int(np.argmax(np.abs(ours) if signed else ours))
            metrics[name] = dict(relative_l2=float(np.sqrt(trap(error**2, times)/ref_norm)) if ref_norm > 0 else None,
                                 rms=float(np.sqrt(trap(error**2, times)/duration)),
                                 max_abs=float(np.max(np.abs(error))),
                                 ours_peak=float(ours[own_peak]), reference_peak=float(values[reference_peak]),
                                 ours_peak_time=float(times[own_peak]), reference_peak_time=float(rt[reference_peak]),
                                 peak_relative_error=float(ours[own_peak]/values[reference_peak]-1) if values[reference_peak] != 0 else None,
                                 peak_time_error=float(times[own_peak]-rt[reference_peak]))
            plot_data[name] = ours
            for i, t in enumerate(times):
                if len(differences) <= i:
                    differences.append(dict(time=float(t)))
                differences[i].update({name:float(ours[i]), name+'_reference':float(expected[i]), name+'_error':float(error[i])})
        initial_mass, initial_energy = history[0]['mass'], history[0]['total_energy']
        summary = dict(directory=str(directory), config=config, final_time=float(times[-1]),
                       comparisons=metrics, diagnostic_samples=len(times),
                       largest_diagnostic_gap=float(np.diff(times).max()),
                       sparse_temporal_sampling=bool(np.diff(times).max() > .15),
                       metric_note='Errors use stored diagnostic times; sparse sampling does not validate curve shape or locate a peak',
                       max_mass_relative_change=max(abs(r['mass']/initial_mass-1) for r in history),
                       max_total_energy_relative_change=max(abs(r['total_energy']/initial_energy-1) for r in history),
                       reference_mach=.1, mach_mismatch=config['mach'] != .1,
                       reference_initialization='constant-density', initialization_mismatch=config['initialization'] != 'constant-density')
        if 'cells' in config:
            h_eff = max(2*math.pi/(n*(config['p']+1)) for n in config['cells'])
            peak_viscous = max(r['epsilon_viscous'] for r in history)
            eta = ((1/config['reynolds'])**3/peak_viscous)**.25
            summary['resolution_estimate'] = dict(effective_spacing=h_eff,
                smallest_mean_kolmogorov_eta=eta, spacing_over_eta=h_eff/eta,
                nominal_kmax_eta=math.pi*eta/h_eff,
                note='From volume-mean viscous dissipation; nominal Nyquist, not a FR resolution guarantee')
        results.append(summary)
        write_csv(output/(directory.name+'-difference.csv'), differences)
        write_csv(output/(directory.name+'-metrics.csv'), [dict(quantity=name, **metric) for name, metric in metrics.items()])
        curves.append((directory.name, config, times, plot_data, history))
    (output/'comparison.json').write_text(json.dumps(dict(reference=str(reference),
             definitions='Match FLEXI column definitions; physical=DR_Sd+DR_p; epsilon_kinetic reference from second-order np.gradient(Ekin comp, Time); temperature normalized by initial mean; entropy measured relative to initial mean', runs=results), indent=2, allow_nan=False)+'\n')
    if plot:
        import os
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/fr-matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        groups = [
            ('reference_comparison', (2, 2), [
                ('kinetic_energy', 'Density-weighted kinetic energy'),
                ('epsilon_kinetic', 'Kinetic energy decay rate -dK/dt'),
                ('epsilon_vorticity', 'Solenoidal dissipation'),
                ('enstrophy', 'Density-weighted enstrophy')]),
            ('dissipation_components', (2, 3), [
                ('epsilon_strain', 'Strain dissipation'), ('epsilon_viscous', 'Deviatoric viscous dissipation'),
                ('epsilon_physical', 'Viscous dissipation - pressure work'),
                ('epsilon_velocity_gradient', 'Velocity-gradient dissipation'),
                ('epsilon_dilatation', 'Dilatational dissipation'),
                ('pressure_dilatation', 'Pressure dilatation <p div u>')]),
            ('additional_statistics', (2, 3), [
                ('kinetic_energy_unweighted', 'Unweighted kinetic energy'), ('max_vorticity', 'Maximum vorticity'),
                ('uprime', 'Velocity RMS scale'), ('mean_temperature_normalized', 'Mean temperature / initial mean'),
                ('entropy_change', 'Mean entropy diagnostic - initial mean')])]
        for stem, shape, panels in groups:
            fig, axes = plt.subplots(*shape, figsize=(5.3*shape[1], 3.8*shape[0]))
            for ax, (name, ylabel) in zip(axes.flat, panels):
                available = [c for c in curves if name in c[3]]
                if not available:
                    ax.set_visible(False)
                    continue
                ref_label = 'FLEXI p=7, 512³ nodes, Ma=0.1'
                if name == 'epsilon_kinetic':
                    ref_label += ' (derivative of published K)'
                ax.plot(rt, reference_curves[name], 'k--', linewidth=1.5, label=ref_label)
                for label, config, times, data, history in available:
                    ax.plot(times, data[name], '-', marker='o', markersize=2, linewidth=1.2,
                            label=f'FR p={config["p"]}, Ma={config["mach"]}')
                ax.set(xlabel='t U0/L', ylabel=ylabel, xlim=(0, max(c[2][-1] for c in available)))
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            for ax in list(axes.flat)[len(panels):]:
                ax.set_visible(False)
            fig.tight_layout()
            fig.savefig(output/(stem+'.png'), dpi=180)
            fig.savefig(output/(stem+'.pdf'))
            plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    result = compare_runs(args.directories, args.output, reference=args.reference, plot=not args.no_plot)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
