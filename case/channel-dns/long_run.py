"""Resumable constant-mass-flux channel DNS using the fixed native fast path."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from channel import Channel
from grouped_fused import GroupedFused
from native_step import NativeLocalRKStepper, NativeMeanAccumulator
from native_build import locality_options
from runtime import configure_cpu_memory_reuse


def process_id():
    # Report the PID visible to this process; it may be namespace-local.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('NSpid:'):
            return int(line.split()[1])
    return os.getpid()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def save_checkpoint(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@torch.no_grad()
def turbulence(model, U, step, t):
    rho, velocity, pressure = model.eq.primitive(U)
    plane = model.plane_mean(velocity)
    fluctuation = velocity-plane[None, :, None, None, :, None, :]
    volume = model.volume
    variance = model.s.integral(fluctuation.square())/volume
    rms = variance.clamp_min(0).sqrt()
    total = model.s.integral(U)
    bulk = float(total[1]/total[0])
    intensity = float((variance.sum()/3).sqrt())/abs(bulk)
    products = torch.cat((fluctuation.square(),
                          (fluctuation[..., 0]*fluctuation[..., 1])[..., None]), -1)
    profile = model.plane_mean(products)
    return dict(step=step, time=t, sampling='instantaneous, before scheduled kick',
        bulk_u=bulk, turbulence_intensity=intensity, component_rms=rms.tolist(),
        component_intensity=(rms/abs(bulk)).tolist(), tke=float(variance.sum()/2),
        mass=float(total[0]), rho_min=float(rho.min()), rho_max=float(rho.max()),
        pressure_min=float(pressure.min()),
        max_mach=float((velocity.square().sum(-1)/(model.eq.gamma*pressure/rho)).sqrt().max()),
        y=model.s.coords[1][0, :, 0, 0, :, 0].tolist(),
        plane_velocity=plane.tolist(), plane_variance_uv=profile.tolist(),
        transition_verified=False)



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cells', type=int, nargs=3, default=[64, 24, 32])
    parser.add_argument('--steps', type=int, default=100000, help='absolute final step, including resumed steps')
    parser.add_argument('--mach', type=float, default=.3, help='Mach based on reference velocity 15')
    parser.add_argument('--cfl', type=float, default=1.2)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--interval', type=int, default=10000)
    parser.add_argument('--perturb-until', type=int, default=30000)
    parser.add_argument('--sample-start-step', type=int, default=60000)
    parser.add_argument('--perturbation', type=float, default=.5)
    args = parser.parse_args()
    if (min(*args.cells, args.steps, args.threads, args.interval) <= 0
            or args.perturb_until < 0 or args.sample_start_step < args.perturb_until
            or not math.isfinite(args.mach) or not 0 < args.mach < 1
            or not math.isfinite(args.cfl) or args.cfl <= 0
            or not math.isfinite(args.perturbation) or args.perturbation < 0):
        parser.error('invalid grid, counts, Mach, CFL or perturbation')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output/'checkpoint.pt'
    if not args.resume and (args.output/'run.json').exists():
        parser.error('output already contains a run; use --resume or another output')
    config = dict(cells=args.cells, p=3, stretch=1.2, re_tau=178.12, mach=args.mach,
        cfl=args.cfl, interval=args.interval, perturb_until=args.perturb_until,
        perturbation=args.perturbation, sample_start_step=args.sample_start_step,
        layout='soa', ldg_backend='fused', convection_backend='flux-local-explicit', rk_backend='add',
        backend='native', convection_block=[4, 4, 4], ldg_block=[4, 4, 4],
        driving='constant_mass_flux', lengths=[4*math.pi, 4*math.pi/3],
        bulk_velocity=15.67873098359455, target_mass_flux=15.67873098359455,
        reference='Moser, Kim & Mansour (1999), chan180.means',
        bulk_target_method='trapezoidal integral of reference U+ over y/h',
        step_backend='group-local-rk-fused-finish', mean_backend='avx2-fused-trapezoid',
        rhs_backend='group-local-convection-ldg', primitive_reuse=False)
    allocator = configure_cpu_memory_reuse(True)
    torch.set_num_threads(args.threads)
    model = Channel(tuple(args.cells), re_tau=config['re_tau'], mach=args.mach, forcing=0.,
                    lengths=config['lengths'], bulk_velocity=config['bulk_velocity'])
    first_y_plus = config['re_tau']*(1+float(model.s.z[0]))*float(model.s.widths[1][0])/2
    if first_y_plus > 1:
        parser.error('nearest wall Gauss point exceeds reference y+=1')
    mean = NativeMeanAccumulator(model)
    if args.resume:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['config'] != config:
            parser.error('checkpoint numerical configuration differs from supplied arguments')
        U, step, t = saved['U'], saved['step'], saved['time']
        field_integral, duration = saved['field_integral'], saved['sample_duration']
        perturbations, diagnostics = saved['perturbations'], saved['diagnostics']
        mass0 = saved['initial_mass']
    else:
        U = model.initial(args.perturbation).movedim(-1, 0).contiguous().movedim(0, -1)
        step, t, duration = 0, 0., 0.
        field_integral = mean.zeros()
        perturbations = [dict(step=0, time=0., amplitude=args.perturbation, kind='initial')]
        diagnostics = []
        mass0 = float(model.s.integral(U)[0])
    plan = GroupedFused(model)
    native = NativeLocalRKStepper(model, plan)
    atomic_json(args.output/'run.json', dict(config=config, target_steps=args.steps,
        threads=args.threads, allocator=allocator, pid=process_id(), native_tuning='reuse-first',
        native_locality=locality_options(), native_fp_contract='fast', native_ldg_halo_pruning=True,
        nearest_gauss_wall_y_plus=first_y_plus, resumed=args.resume))
    started, start_step = time.perf_counter(), step
    checkpoint_step = step if args.resume else None

    def status(state):
        elapsed = time.perf_counter()-started
        atomic_json(args.output/'status.json', dict(status=state, pid=process_id(),
            step=step, target_steps=args.steps, time=t, cfl=args.cfl, elapsed_seconds=elapsed,
            seconds_per_step=elapsed/(step-start_step) if step > start_step else None,
            sample_duration=duration, checkpoint_step=checkpoint_step,
            pressure_gradient_force=float(native.last_force) if native.last_force is not None else None,
            latest_diagnostic=diagnostics[-1] if diagnostics else None))

    def checkpoint():
        save_checkpoint(checkpoint_path, dict(version=1, config=config, U=U, step=step, time=t,
            field_integral=field_integral, sample_duration=duration, has_previous_moments=False,
            perturbations=perturbations, diagnostics=diagnostics, initial_mass=mass0))

    status('starting')
    print(f'start pid={process_id()} step={step} target={args.steps} CFL={args.cfl} threads={args.threads}', flush=True)
    next_dt = None
    try:
        while step < args.steps:
            if next_dt is None:
                valid, next_dt = native.scan(U, args.cfl)
                if not valid:
                    raise RuntimeError(f'nonphysical state on step {step}')
            dt = next_dt
            if not math.isfinite(dt) or dt <= 0 or t+dt == t:
                raise RuntimeError('invalid time step')
            sampling = step >= args.sample_start_step
            candidate, valid, next_dt = native.advance_finish(U, t, dt, args.cfl,
                                                             field_integral if sampling else None)
            if not valid:
                raise RuntimeError(f'nonphysical state on step {step+1}')
            if sampling:
                duration += dt
            U, t, step = candidate, t+dt, step+1
            if step % args.interval == 0:
                diagnostic = turbulence(model, U, step, t)
                diagnostic['mass_relative_change'] = diagnostic['mass']/mass0-1
                flux = float(model.s.integral(U[..., 1:2])[0])/model.volume
                diagnostic.update(mass_flux=flux, mass_flux_relative_error=flux/config['target_mass_flux']-1,
                                  pressure_gradient_force=float(native.last_force))
                diagnostics.append(diagnostic)
                atomic_json(args.output/'turbulence.json', diagnostics)
                print(f'turbulence step={step} time={t:.8g} intensity={diagnostic["turbulence_intensity"]:.8g} rms={diagnostic["component_rms"]}', flush=True)
                if step <= args.perturb_until:
                    U = model.perturb(U, args.perturbation, step//args.interval)
                    next_dt = None
                    perturbations.append(dict(step=step, time=t, amplitude=args.perturbation, kind='kick'))
                    print(f'perturbation injected step={step}', flush=True)
                checkpoint()
                checkpoint_step = step
                print(f'checkpoint replaced step={step}', flush=True)
            if step % 100 == 0 or (step <= 100 and step % 10 == 0):
                status('running')
                print(f'progress step={step} time={t:.8g} dt={dt:.7g}', flush=True)
        if checkpoint_step != step:
            checkpoint()
            checkpoint_step = step
        status('complete')
    except BaseException as error:
        status('failed')
        atomic_json(args.output/'failure.json', dict(step=step, time=t, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
