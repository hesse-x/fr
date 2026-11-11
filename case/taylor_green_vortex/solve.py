"""Taylor–Green vortex using the shared Cartesian NS execution presets."""
import argparse
import csv
import json
import math
from pathlib import Path
import time

import torch
from fr import CartesianNavierStokes


class TaylorGreenVortex(CartesianNavierStokes):
    """rho0=U0=L=R=1; Ma changes background pressure, never Re or velocity."""
    def __init__(self, cells=(64, 64, 64), p=3, *, reynolds=1600., mach=.3,
                 gamma=1.4, prandtl=.71, initialization='constant-density', device='cpu', anti_alias='none',
                 quadrature_order=None):
        if not math.isfinite(reynolds) or reynolds <= 0:
            raise ValueError('positive finite Reynolds number required')
        if not math.isfinite(mach) or not 0 < mach < 1:
            raise ValueError('require 0 < reference Mach < 1')
        if initialization not in ('constant-density', 'constant-temperature'):
            raise ValueError('unknown initialization')
        self.initialization = initialization
        self.reynolds, self.mach = reynolds, mach
        self.p0 = 1/(gamma*mach*mach)
        super().__init__(cells, ((-math.pi, math.pi),)*3, p, mu=1/reynolds,
                         gamma=gamma, prandtl=prandtl, device=device,
                         boundary='periodic', forcing=0., anti_alias=anti_alias,
                         quadrature_order=quadrature_order,
                         reference_state=[1., 0., 0., 0., self.p0/(gamma-1)])

    def initial_at(self, coords):
        x, y, z = torch.broadcast_tensors(*coords)
        velocity = torch.stack((torch.sin(x)*torch.cos(y)*torch.cos(z),
                                -torch.cos(x)*torch.sin(y)*torch.cos(z),
                                torch.zeros_like(z)), -1)
        pressure = self.p0+(torch.cos(2*x)+torch.cos(2*y))*(torch.cos(2*z)+2)/16
        density = torch.ones_like(pressure) if self.initialization == 'constant-density' else pressure/self.p0
        return self.eq.conservative(density, velocity, pressure)

    def initial(self):
        return self.soa(self.initial_at(self.s.coords))

    def flow_fields(self, U, *, include_q=True):
        rho, velocity, pressure = self.eq.primitive(U)
        primitive = self.primitive(U)
        gradient = self.ldg.gradient(primitive)
        du = gradient[..., :3, :]
        omega = torch.stack((du[..., 2, 1]-du[..., 1, 2],
                             du[..., 0, 2]-du[..., 2, 0],
                             du[..., 1, 0]-du[..., 0, 1]), -1)
        q = -.5*(du*du.transpose(-1, -2)).sum((-1, -2)) if include_q else None
        return rho, velocity, pressure, gradient, omega, q

    def diagnostics(self, U, t=0., *, rhs=None):
        rho, velocity, pressure, gradient, omega, q = self.flow_fields(U, include_q=False)
        def avg(value):
            return self.s.integral(value[..., None])[0]/self.volume
        speed2 = velocity.square().sum(-1)
        rhs = self.rhs(U, t) if rhs is None else rhs
        kinetic_rate = avg((velocity*rhs[..., 1:4]).sum(-1)-.5*speed2*rhs[..., 0])
        du = gradient[..., :3, :]
        divergence = du.diagonal(dim1=-2, dim2=-1).sum(-1)
        strain = .5*(du+du.transpose(-1, -2))
        strain2 = strain.square().sum((-1, -2))
        viscous = 2*self.mu*avg(strain2-divergence.square()/3)
        pressure_work = avg(pressure*divergence)
        enstrophy = avg(.5*rho*omega.square().sum(-1))
        unweighted = avg(.5*omega.square().sum(-1))
        total = self.s.integral(U)/self.volume
        mean_velocity = torch.stack([avg(velocity[..., a]) for a in range(3)])
        variances = torch.stack([avg((velocity[..., a]-mean_velocity[a]).square()) for a in range(3)])
        result = dict(time=float(t), kinetic_energy=float(avg(.5*rho*speed2)),
                      kinetic_energy_unweighted=float(avg(.5*speed2)),
                      turbulent_kinetic_energy=float(.5*variances.sum()),
                      favre_turbulent_kinetic_energy=float(avg(.5*rho*speed2)-.5*total[1:4].square().sum()/total[0]),
                      mean_u=float(mean_velocity[0]), mean_v=float(mean_velocity[1]), mean_w=float(mean_velocity[2]),
                      rms_u=float(variances[0].sqrt()), rms_v=float(variances[1].sqrt()), rms_w=float(variances[2].sqrt()),
                      epsilon_strain=float(2*self.mu*avg(strain2)),
                      dkinetic_dt=float(kinetic_rate), epsilon_kinetic=float(-kinetic_rate),
                      enstrophy=float(enstrophy), enstrophy_unweighted=float(unweighted),
                      epsilon_enstrophy=float(2*self.mu*enstrophy),
                      epsilon_vorticity=float(2*self.mu*unweighted),
                      epsilon_viscous=float(viscous), pressure_dilatation=float(pressure_work),
                      epsilon_velocity_gradient=float(self.mu*avg(du.square().sum((-1, -2)))),
                      epsilon_dilatation=float(4/3*self.mu*avg(divergence.square())),
                      max_vorticity=float(omega.square().sum(-1).sqrt().max()),
                      mean_temperature=float(avg(pressure/rho)),
                      mean_temperature_normalized=float(avg(pressure/rho)/self.p0),
                      mean_entropy=float(avg(-rho/(self.eq.gamma-1)*(pressure.log()-self.eq.gamma*rho.log()))),
                      uprime=float((2/3*avg(.5*speed2)).sqrt()),
                      kinetic_budget_residual=float(-kinetic_rate-viscous+pressure_work),
                      mass=float(total[0]), momentum_x=float(total[1]), momentum_y=float(total[2]),
                      momentum_z=float(total[3]), total_energy=float(total[4]),
                      rho_min=float(rho.min()), rho_max=float(rho.max()),
                      density_rms=float(avg((rho-avg(rho)).square()).sqrt()),
                      pressure_min=float(pressure.min()),
                      max_mach=float((speed2/(self.eq.gamma*pressure/rho)).sqrt().max()))
        if not all(math.isfinite(v) for v in result.values()):
            raise RuntimeError('nonfinite diagnostic')
        return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def write_history(path, history):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


@torch.no_grad()
def run(output, *, cells=(64, 64, 64), p=3, reynolds=1600., mach=.3,
        gamma=1.4, prandtl=.71, initialization='constant-density', backend='native-p3', device='cpu',
        anti_alias='none', quadrature_order=None, final_time=10., cfl=.8,
        max_dt=None, max_steps=None, diagnostic_interval=100, checkpoint_interval=10000,
        snapshot_times=(10.,), sample_points=None, fields=True, plot=True, resume=False, compare_with=()):
    output = Path(output)
    if not math.isfinite(final_time) or final_time < 0:
        raise ValueError('finite nonnegative final_time required')
    if not math.isfinite(cfl) or cfl <= 0 or (max_dt is not None and (not math.isfinite(max_dt) or max_dt <= 0)):
        raise ValueError('positive finite CFL and optional max_dt required')
    if diagnostic_interval < 1 or checkpoint_interval < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError('positive intervals and optional max_steps required')
    if any(not math.isfinite(t) or t < 0 for t in snapshot_times):
        raise ValueError('finite nonnegative snapshot times required')
    config = dict(cells=list(cells), p=p, reynolds=reynolds, mach=mach, gamma=gamma,
                  prandtl=prandtl, initialization=initialization, backend=backend, anti_alias=anti_alias,
                  quadrature_order=quadrature_order, cfl=cfl, max_dt=max_dt)
    model = TaylorGreenVortex(tuple(cells), p, reynolds=reynolds, mach=mach,
                             gamma=gamma, prandtl=prandtl, initialization=initialization, device=device,
                             anti_alias=anti_alias, quadrature_order=quadrature_order)
    stepper = model.make_stepper(backend, constant_mass_flux=False)
    checkpoint = output/'checkpoint.pt'
    if resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['config'] != config:
            raise ValueError('checkpoint numerical configuration differs')
        U, t, step = model.soa(saved['U']), saved['time'], saved['step']
        history, snapshots = saved['history'], saved['snapshots']
        if final_time < t:
            raise ValueError('final_time precedes checkpoint')
    else:
        if checkpoint.exists() or (output/'run.json').exists():
            raise ValueError('output already contains a run; use resume or another directory')
        U, t, step, history, snapshots = model.initial(), 0., 0, [], []
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'run.json', dict(config=config, target_time=final_time,
                periodic=list(model.s.periodic), constant_mass_flux=False,
                threads=torch.get_num_threads(), diagnostic_interval=diagnostic_interval,
                checkpoint_interval=checkpoint_interval, snapshot_times=list(snapshot_times),
                compare_with=[str(Path(path)) for path in compare_with],
                normalization='rho0=U0=L=R=1; time=t*U0/L; volume means',
                implementation=stepper.plan.metadata if backend == 'native-p3' else 'torch reference'))
    rhs = stepper.plan if backend == 'native-p3' else model.rhs
    started = time.perf_counter()
    next_dt = None
    targets = sorted(set(snapshot_times))

    def update_comparison():
        if len(history) >= 2 and reynolds == 1600. and gamma == 1.4 and prandtl == .71:
            from .postprocess import compare_runs
            compare_runs([*map(Path, compare_with), output], output/'comparison', plot=plot)

    def observe():
        if history and history[-1]['step'] == step:
            return
        row = model.diagnostics(U, t, rhs=rhs(U, t))
        row.update(step=step, elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        write_history(output/'history.csv', history)
        atomic_json(output/'status.json', dict(status='running', step=step, time=t,
                    target_time=final_time, elapsed_seconds=time.perf_counter()-started,
                    last_diagnostic=row))
        print(f'step={step} t={t:.6f} K={row["kinetic_energy"]:.8g} '
              f'eps={row["epsilon_kinetic"]:.7g} rho=[{row["rho_min"]:.6g},{row["rho_max"]:.6g}]', flush=True)
        if step > 0 and step % 500 == 0:
            update_comparison()

    def snapshot():
        if any(abs(t-v) < 1e-12 for v in snapshots):
            return
        from .postprocess import export_snapshot
        export_snapshot(model, U, t, output/f'snapshot-{t:08.4f}',
                        sample_points=sample_points, fields=fields, plot=plot)
        snapshots.append(t)

    def commit(status):
        data = dict(version=1, config=config, U=U, time=t, step=step,
                    history=history, snapshots=snapshots)
        save_checkpoint(checkpoint, data)
        if step % checkpoint_interval == 0 or status == 'complete':
            save_checkpoint(output/f'checkpoint-{step:08d}.pt', data)
        atomic_json(output/'status.json', dict(status=status, step=step, time=t,
                    target_time=final_time, elapsed_seconds=time.perf_counter()-started,
                    last_diagnostic=history[-1] if history else None))
        update_comparison()

    observe()
    if any(abs(t-v) < 1e-12 for v in targets):
        snapshot()
    try:
        while t < final_time and (max_steps is None or step < max_steps):
            if next_dt is None:
                if backend == 'native-p3':
                    valid, next_dt = stepper.scan(U, cfl)
                    if not valid:
                        raise RuntimeError('invalid state before step')
                else:
                    if not model.eq.valid(U):
                        raise RuntimeError('invalid state before step')
                    next_dt = model.timestep(U, cfl)
            future = [v for v in targets if v > t+1e-12]
            endpoint = min(final_time, future[0]) if future else final_time
            dt = min(next_dt, endpoint-t, max_dt if max_dt is not None else math.inf)
            if not math.isfinite(dt) or dt <= 0 or t+dt == t:
                raise RuntimeError('invalid time step')
            if backend == 'native-p3':
                candidate, valid, next_dt = stepper.advance_finish(U, t, dt, cfl)
            else:
                candidate = stepper(U, t, dt)
                valid, next_dt = model.eq.valid(candidate), None
            if not valid:
                raise RuntimeError(f'nonphysical state at step {step+1}; previous checkpoint retained')
            U, t, step = candidate, min(t+dt, endpoint), step+1
            at_snapshot = any(abs(t-v) < 1e-12 for v in targets)
            if step % diagnostic_interval == 0 or at_snapshot or t == final_time:
                observe()
            if at_snapshot:
                snapshot()
            if at_snapshot or step % checkpoint_interval == 0:
                if t != final_time:
                    commit('running')
            elif step % 100 == 0:
                atomic_json(output/'status.json', dict(status='running', step=step, time=t,
                            target_time=final_time, elapsed_seconds=time.perf_counter()-started,
                            last_diagnostic=history[-1] if history else None))
                print(f'progress step={step} t={t:.6f} dt={dt:.7g}', flush=True)
        observe()
        commit('complete' if t == final_time else 'step-limit')
    except BaseException as error:
        atomic_json(output/'failure.json', dict(step=step, time=t, error=repr(error)))
        raise
    return U, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--cells', nargs=3, type=int)
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--mach', type=float, default=.3)
    parser.add_argument('--reynolds', type=float, default=1600.)
    parser.add_argument('--initialization', choices=('constant-density', 'constant-temperature'), default='constant-density')
    parser.add_argument('--backend', choices=('native-p3', 'torch'), default='native-p3')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--anti-alias', default='none')
    parser.add_argument('--quadrature-order', type=int)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--time', type=float, default=10.)
    parser.add_argument('--cfl', type=float, default=.8)
    parser.add_argument('--max-dt', type=float)
    parser.add_argument('--steps', type=int, help='absolute step limit; save a resumable checkpoint')
    parser.add_argument('--diagnostic-interval', type=int, default=100)
    parser.add_argument('--checkpoint-interval', type=int, default=10000)
    parser.add_argument('--snapshot-times', type=float, nargs='*', default=[10.])
    parser.add_argument('--sample-points', type=int)
    parser.add_argument('--no-fields', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--compare-with', type=Path, nargs='*', default=[], help='include previous runs in reference comparisons')
    parser.add_argument('--output', type=Path, default=Path('results/taylor_green_vortex/ma03-n64-p3-t10-stats100'))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('positive threads required')
    torch.set_num_threads(args.threads)
    if args.device == 'cpu':
        from fr.native.cartesian_ns.runtime import configure_cpu_memory_reuse
        configure_cpu_memory_reuse()
    run(args.output, cells=tuple(args.cells or (args.n,)*3), p=args.p, reynolds=args.reynolds,
        mach=args.mach, initialization=args.initialization, backend=args.backend, device=args.device, anti_alias=args.anti_alias,
        quadrature_order=args.quadrature_order, final_time=args.time, cfl=args.cfl,
        max_dt=args.max_dt, max_steps=args.steps, diagnostic_interval=args.diagnostic_interval,
        checkpoint_interval=args.checkpoint_interval, snapshot_times=args.snapshot_times,
        sample_points=args.sample_points, fields=not args.no_fields, plot=not args.no_plot, resume=args.resume, compare_with=args.compare_with)


if __name__ == '__main__':
    main()
