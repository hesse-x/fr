"""Resumable constant-mass-flux channel DNS using the fixed native fast path."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
Channel = import_module('case.channel-dns.channel').Channel
from fr.native.cartesian_ns import NativeMeanAccumulator
from fr.native.cartesian_ns.native_build import locality_options
from fr.native.cartesian_ns.runtime import configure_cpu_memory_reuse


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


def _log_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


@torch.no_grad()
def state_failure_details(model, U):
    """Inspect only on fatal failure; preserve raw pressure, including invalid values."""
    rho = U[..., 0]
    kinetic = .5*U[..., 1:4].square().sum(-1)/rho
    pressure = (model.eq.gamma-1)*(U[..., 4]-kinetic)
    result = {}
    masks = dict(nonpositive_density=rho <= 0, nonpositive_pressure=pressure <= 0,
                 nonfinite_density=~torch.isfinite(rho),
                 nonfinite_pressure=~torch.isfinite(pressure),
                 nonfinite_conservative=~torch.isfinite(U).all(-1))
    for name, field in (('density', rho), ('pressure', pressure)):
        finite = field[torch.isfinite(field)]
        result[name+'_finite_min'] = float(finite.min()) if finite.numel() else None
        result[name+'_finite_max'] = float(finite.max()) if finite.numel() else None
    result['counts'] = {name: int(mask.sum()) for name, mask in masks.items()}
    examples = {}
    for name, mask in masks.items():
        if not result['counts'][name]:
            continue
        index = tuple(int(i) for i in torch.nonzero(mask, as_tuple=False)[0])
        examples[name] = dict(cell=list(index[:3]), node=list(index[3:]),
            coordinates=[float(coord[index]) for coord in model.s.coords],
            density=_log_number(rho[index]), pressure=_log_number(pressure[index]),
            total_energy=_log_number(U[index][4]), kinetic_energy=_log_number(kinetic[index]),
            conservative=[_log_number(v) for v in U[index]])
    result['examples'] = examples
    return result


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
def main(*, model_class=Channel, configure_parser=None, model_options=None,
         output_default=None, default_mesh=None, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output', type=Path, default=output_default, required=output_default is None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--failure-history-steps', type=int, default=5,
                        help='recent accepted states retained in memory and dumped on fatal failure; 0 disables')
    parser.add_argument('--mesh', type=Path, help='Cartesian JSON mesh; overrides cell counts')
    parser.add_argument('--cells', type=int, nargs=3)
    parser.add_argument('--steps', type=int, default=100000, help='absolute final step, including resumed steps')
    parser.add_argument('--re-tau', type=float, default=178.12)
    parser.add_argument('--mach', type=float, default=.3, help='Mach based on reference velocity 15')
    parser.add_argument('--cfl', type=float, default=1.2)
    parser.add_argument('--viscous-scheme', choices=('ldg', 'br1-penalty'), default='ldg')
    parser.add_argument('--penalty', type=float, default=4.)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--interval', type=int, default=10000)
    parser.add_argument('--perturb-until', type=int, default=30000)
    parser.add_argument('--sample-start-step', type=int, default=60000)
    parser.add_argument('--perturbation', type=float, default=.5)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args()
    extra_options = {} if model_options is None else model_options(args)
    if args.mesh is None and args.cells is None:
        args.mesh = default_mesh or Path(__file__).with_name('mesh')/'default.json'
    args.cells = args.cells or [64, 24, 32]
    if (min(*args.cells, args.steps, args.threads, args.interval) <= 0
            or args.failure_history_steps < 0
            or args.perturb_until < 0 or args.sample_start_step < args.perturb_until
            or not math.isfinite(args.re_tau) or args.re_tau <= 0
            or not math.isfinite(args.mach) or not 0 < args.mach < 1
            or not math.isfinite(args.cfl) or args.cfl <= 0
            or not math.isfinite(args.penalty) or args.penalty <= 0
            or not math.isfinite(args.perturbation) or args.perturbation < 0):
        parser.error('invalid grid, counts, Mach, CFL or perturbation')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output/'checkpoint.pt'
    if not args.resume and (args.output/'run.json').exists():
        parser.error('output already contains a run; use --resume or another output')
    if args.mesh is not None:
        from fr import CartesianMesh
        args.cells = list(CartesianMesh.read(args.mesh).cells)
    config = dict(cells=args.cells, p=3, stretch=1.2, re_tau=args.re_tau, mach=args.mach,
        cfl=args.cfl, interval=args.interval, perturb_until=args.perturb_until,
        perturbation=args.perturbation, sample_start_step=args.sample_start_step,
        layout='soa', ldg_backend='fused', convection_backend='flux-local-explicit', rk_backend='add',
        backend='native', convection_block=[4, 4, 4], ldg_block=[4, 4, 4],
        driving='constant_mass_flux', lengths=[4*math.pi, 4*math.pi/3],
        bulk_velocity=15.67873098359455, target_mass_flux=15.67873098359455,
        reference='Moser, Kim & Mansour (1999), chan180.means',
        bulk_target_method='trapezoidal integral of reference U+ over y/h',
        step_backend='group-local-rk-fused-finish', mean_backend='avx2-fused-trapezoid',
        rhs_backend='group-local-convection-ldg' if args.viscous_scheme=='ldg' else 'group-local-convection-br1-penalty',
        primitive_reuse=False, viscous_scheme=args.viscous_scheme, penalty=args.penalty)
    config.update(extra_options)
    if extra_options.get('modal_filter') == 'hifiles':
        config.update(step_backend='group-local-rk-filtered',
                      mean_backend='native-trapezoid-after-filter')
    if args.mesh is not None:
        config['mesh_edges'] = [e.tolist() for e in CartesianMesh.read(args.mesh).edges]
    allocator = configure_cpu_memory_reuse(True)
    torch.set_num_threads(args.threads)
    model = model_class(tuple(args.cells), mesh=args.mesh, re_tau=config['re_tau'], mach=args.mach, forcing=0.,
                    lengths=config['lengths'], bulk_velocity=config['bulk_velocity'],
                    viscous_scheme=args.viscous_scheme, penalty=args.penalty, **extra_options)
    first_y_plus = config['re_tau']*(1+float(model.s.z[0]))*float(model.s.widths[1][0])/2
    wall_limit = getattr(args, 'max_wall_y_plus', 1.)
    if not math.isfinite(wall_limit) or wall_limit <= 0:
        parser.error('positive finite maximum wall y+ required')
    if first_y_plus > wall_limit:
        parser.error(f'nearest wall Gauss point exceeds reference y+={wall_limit:g}')
    mean = NativeMeanAccumulator(model)
    if args.resume:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        saved_config = saved['config'].copy()
        saved_config.setdefault('viscous_scheme', 'ldg')
        saved_config.setdefault('penalty', 4.)
        if 'modal_filter' in extra_options:
            for key, value in dict(modal_filter='none', filter_threshold=.1,
                                   filter_alpha=36., filter_order=2).items():
                saved_config.setdefault(key, value)
        # An explicit new sampling schedule is safe only before any sampling
        # has begun; allow starting at this checkpoint or a later step.
        if (saved['sample_duration'] == 0
                and saved_config['sample_start_step'] > saved['step']
                and args.sample_start_step >= saved['step']):
            saved_config['sample_start_step'] = args.sample_start_step
        if saved_config != config:
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
    native = model.make_stepper('native-p3', constant_mass_flux=True)
    plan = native.plan
    atomic_json(args.output/'run.json', dict(config=config, target_steps=args.steps,
        threads=args.threads, allocator=allocator, pid=process_id(), native_tuning='reuse-first',
        native_locality=locality_options(), native_fp_contract='fast',
        native_ldg_halo_pruning=plan.metadata['dependency_pruned_halo'],
        native_preset='native-p3', native_boundary_mode=plan.metadata['boundary_mode'],
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
            latest_diagnostic=diagnostics[-1] if diagnostics else None,
            modal_filter=native.modal_filter.summary() if getattr(native, 'modal_filter', None) is not None else None))

    def checkpoint():
        save_checkpoint(checkpoint_path, dict(version=1, config=config, U=U, step=step, time=t,
            field_integral=field_integral, sample_duration=duration, has_previous_moments=False,
            perturbations=perturbations, diagnostics=diagnostics, initial_mass=mass0))

    status('starting')
    print(f'start pid={process_id()} step={step} target={args.steps} CFL={args.cfl} threads={args.threads}', flush=True)
    next_dt = None
    failure_details = {}
    recent_states = deque(maxlen=args.failure_history_steps)
    if recent_states.maxlen:
        recent_states.append(dict(step=step, time=t, dt=None, U=U))

    def report_invalid(state, failed_step, stage, dt=None):
        details = dict(failed_step=failed_step, stage=stage, time=t, dt=dt, cfl=args.cfl,
            pressure_gradient_force=_log_number(native.last_force) if native.last_force is not None else None,
            invalid_state=state_failure_details(model, state))
        if state is not U:
            details['previous_state'] = state_failure_details(model, U)
        if recent_states.maxlen:
            history_path = args.output/'failure_history.pt'
            save_checkpoint(history_path, dict(config=config, edges=model.s.edges,
                history=list(recent_states), failed_step=failed_step, failed_time=t,
                dt=dt, failed_stage=stage, failed_state=state))
            details['history_file'] = history_path.name
            details['history_steps'] = [entry['step'] for entry in recent_states]
        failure_details.update(details)
        print('FATAL nonphysical state: '+json.dumps(details, allow_nan=False),
              file=sys.stderr, flush=True)
        return details

    try:
        while step < args.steps:
            if next_dt is None:
                valid, next_dt = native.scan(U, args.cfl)
                if not valid:
                    report_invalid(U, step, 'input scan')
                    raise RuntimeError(f'nonphysical state on step {step}')
            dt = next_dt
            if not math.isfinite(dt) or dt <= 0 or t+dt == t:
                raise RuntimeError('invalid time step')
            sampling = step >= args.sample_start_step
            candidate, valid, next_dt = native.advance_finish(U, t, dt, args.cfl,
                                                             field_integral if sampling else None)
            if not valid:
                report_invalid(candidate, step+1, 'completed SSPRK43 output', dt)
                raise RuntimeError(f'nonphysical state on step {step+1}; see fatal state diagnostics')
            if sampling:
                duration += dt
            U, t, step = candidate, t+dt, step+1
            if recent_states.maxlen:
                recent_states.append(dict(step=step, time=t, dt=dt, U=U))
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
        filter_backend = getattr(native, 'modal_filter', None)
        failed_filter_state = getattr(filter_backend, 'failed_state', None)
        if not failure_details and failed_filter_state is not None:
            report_invalid(failed_filter_state, step+1, 'RK modal filter', dt)
        status('failed')
        atomic_json(args.output/'failure.json', dict(step=step, time=t, error=repr(error)) | failure_details)
        raise


if __name__ == '__main__':
    main()
