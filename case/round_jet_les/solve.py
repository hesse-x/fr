"""Open-domain FR/LDG round jet with WALE LES."""
import argparse
from functools import wraps
import math
from pathlib import Path
import time

import torch

from fr import CartesianMesh
from fr.les import CartesianLESMixin
from fr.modal_filter import CartesianModalFilter
from fr.navier_stokes import CartesianNavierStokes

from .generate_mesh import NOZZLE_RADIUS, build_mesh
from .postprocess import JetSamples, write_velocity_slice


class RoundJetLES(CartesianLESMixin, CartesianNavierStokes):
    """Scales are nozzle diameter, jet speed and ambient density."""

    def __init__(self, mesh, p=3, *, reynolds=6e5, mach=.5, shear_width=.12,
                 disturbance=.03, interface_flux='rusanov', modal_filter='hifiles',
                 outlet='characteristic'):
        if any(not math.isfinite(v) or v <= 0 for v in (reynolds, mach, shear_width)):
            raise ValueError('positive finite Reynolds number, Mach number and shear width required')
        if not math.isfinite(disturbance) or disturbance < 0:
            raise ValueError('finite nonnegative disturbance required')
        if (len(mesh.bounds) != 3 or mesh.bounds[0][0] != 0.
                or any(lo >= -NOZZLE_RADIUS or hi <= NOZZLE_RADIUS for lo, hi in mesh.bounds[1:])):
            raise ValueError('jet mesh must start at x=0 and enclose the nozzle in both transverse directions')
        if modal_filter not in ('none', 'hifiles'):
            raise ValueError('modal filter must be none or hifiles')
        if outlet not in ('characteristic', 'nscbc'):
            raise ValueError('outlet must be characteristic or nscbc')
        self.mach = mach
        self.pressure = 1/(1.4*mach**2)
        self.shear_width = shear_width
        self.disturbance = disturbance
        self.outlet = outlet
        super().__init__(mesh.cells, mesh.bounds, p, edges=mesh.edges, mu=1/reynolds,
                         sgs_model='wale', interface_flux=interface_flux,
                         reference_state=(1., 0., 0., 0., self.pressure/(1.4-1)),
                         boundary='open', inviscid_boundary=self._inviscid_boundary,
                         viscous_boundary=self._viscous_boundary)
        self.modal_filter_name = modal_filter
        self.modal_filter = CartesianModalFilter(self.s, .1, 36., 2) if modal_filter == 'hifiles' else None
        if outlet == 'nscbc':
            convection = self.convection

            @wraps(convection)
            def with_acoustic_outlet(state, t=0., fields=None):
                result = convection(state, t, fields)
                grid = self.s
                trace = grid._trace(state[-1:], grid.R)[0]
                gradient = grid._trace(state[-1:], grid.R @ grid.D)[0]
                correction = self._outlet_flux_correction(trace, gradient)
                lift = grid.gr.reshape(1, 1, -1, 1, 1, 1)
                result[-1] -= (2/grid.widths[0][-1])*lift*correction.unsqueeze(2)
                return result

            self.convection = with_acoustic_outlet

    def initial(self):
        x = self.s.coords[0]
        velocity = x.new_zeros((*x.shape, 3))
        return self.eq.conservative(1., velocity, self.pressure)

    def native_round_jet_data(self):
        _, y, z = self.s.face_coords[0][0]
        parameters = y.new_tensor((self.pressure, self.shear_width, self.disturbance, NOZZLE_RADIUS,
                                   self.outlet == 'nscbc', float(1/(self.s.R @ self.s.gr))))
        return torch.cat((parameters, self.s.R @ self.s.D, torch.stack((y, z), dim=2).reshape(-1)))

    def _outlet_flux_correction(self, state, gradient):
        rho, velocity, pressure = self.eq.primitive(state)
        u = velocity[..., 0]
        gamma = self.eq.gamma
        sound = (gamma*pressure/rho).sqrt()
        speed2 = velocity.square().sum(-1)
        drho, dm, de = gradient[..., 0], gradient[..., 1:4], gradient[..., 4]
        dp = (gamma-1)*(de-(velocity*dm).sum(-1)+.5*speed2*drho)
        # Cancel the incoming normal acoustic amplitude at the outlet trace.
        amplitude = (u-sound)*(dp-sound*(dm[..., 0]-u*drho))/(2*sound.square())
        enthalpy = (state[..., 4]+pressure)/rho
        eigenvector = torch.stack((torch.ones_like(u), u-sound, velocity[..., 1],
                                   velocity[..., 2], enthalpy-sound*u), -1)
        correction = -(amplitude/(self.s.R @ self.s.gr))[..., None]*eigenvector
        # Advective backflow brings in ambient entropy and tangential velocity.
        backflow_velocity = torch.stack((u, torch.zeros_like(u), torch.zeros_like(u)), -1)
        backflow = self.eq.conservative((pressure/self.pressure)**(1/gamma), backflow_velocity, pressure)
        correction += u.clamp_max(0.)[..., None]*(backflow-state)
        return torch.where((u.abs() < sound)[..., None], correction, 0.)

    def nozzle(self, coords, t):
        _, y, z = coords
        radius = torch.hypot(y, z)
        transition = torch.tanh((radius-NOZZLE_RADIUS)/self.shear_width)
        ramp = -math.expm1(-(t/.35)**2)
        angle = torch.atan2(z, y)
        ring = (1-transition.square())*radius/NOZZLE_RADIUS
        perturb = self.disturbance*ramp*ring
        phase = 2*math.pi*t
        velocity = torch.stack((
            .5*(1-transition)*ramp+perturb*torch.cos(4*angle-phase),
            perturb*torch.sin(3*angle+phase),
            perturb*torch.cos(5*angle-1.7*phase),
        ), -1)
        return velocity, .5*(1-transition)

    def _inviscid_boundary(self, coords, inside, axis, side, t):
        density, velocity, pressure = self.eq.primitive(inside)
        gamma = self.eq.gamma
        sound = (gamma*pressure/density).sqrt()
        normal_speed = side*velocity[..., axis]
        if axis == 0 and side == -1:
            target_velocity, _ = self.nozzle(coords, t)
            target_density = torch.ones_like(density)
        elif axis == 0:
            outgoing = normal_speed[..., None] >= 0
            target_velocity = torch.where(outgoing, velocity, 0.)
            target_density = torch.where(outgoing[..., 0], density, 1.)
        else:
            target_velocity = torch.zeros_like(velocity)
            target_density = torch.ones_like(density)
        target_speed = side*target_velocity[..., axis]
        target_sound = (gamma*self.pressure/target_density).sqrt()

        # Take the outgoing acoustic invariant from the interior, the incoming one
        # from the exterior. Entropy and tangential velocity follow the flow.
        scale = 2/(gamma-1)
        plus = normal_speed+scale*sound
        minus = target_speed-scale*target_sound
        boundary_speed = (plus+minus)/2
        boundary_sound = (plus-minus)/(2*scale)
        inflow = boundary_speed < 0
        entropy = torch.where(inflow, self.pressure/target_density**gamma, pressure/density**gamma)
        boundary_density = (boundary_sound.square()/(gamma*entropy))**(1/(gamma-1))
        boundary_velocity = torch.where(inflow[..., None], target_velocity, velocity)
        boundary_velocity[..., axis] = side*boundary_speed
        state = self.eq.conservative(boundary_density, boundary_velocity,
                                     entropy*boundary_density**gamma)
        if self.outlet == 'nscbc' and axis == 0 and side == 1:
            state = torch.where((normal_speed.abs() < sound)[..., None], inside, state)
        target = self.eq.conservative(target_density, target_velocity, self.pressure)
        state = torch.where((normal_speed >= sound)[..., None], inside, state)
        return torch.where((normal_speed <= -sound)[..., None], target, state)

    def _viscous_boundary(self, coords, inside, axis, side, t):
        if (axis, side) != (0, -1):
            return inside
        velocity, weight = self.nozzle(coords, t)
        target = torch.cat((velocity, torch.full_like(weight[..., None], self.pressure)), -1)
        return torch.lerp(inside, target, weight[..., None])


@torch.no_grad()
def run(model, output, *, final_time, cfl, sample_from, sample_interval,
        max_steps=None, plot=True, backend='native-p3', group_packet=None,
        checkpoint_every=0, resume=False, sample_from_step=None, reset_samples=False,
        full_field_statistics=False, sample_max_steps=None, sample_weighting='time',
        slice_every=0, fwh_surface=None, fwh_interval=.04,
        perturb_at_step=(), perturb_amplitude=.005):
    if sample_weighting not in ('time', 'step') or (sample_weighting == 'step' and sample_max_steps != 1):
        raise ValueError('equal-weight sampling requires a one-step sampling interval')
    equal_weight = sample_weighting == 'step'
    if reset_samples and (not resume or sample_from_step is None):
        raise ValueError('resetting samples requires resume with a sampling start step')
    if sample_max_steps is not None and (sample_max_steps < 1 or sample_from_step is None
                                        or sample_from is not None
                                        or sample_interval is None or not 0 < sample_interval < math.inf):
        raise ValueError('a sampling step cap requires a start step and a positive time interval')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'checkpoint.pt'
    if group_packet is None:
        group_packet = 2 if backend in ('native-p3', 'native-hex-p3') else 0
    mapped = backend == 'native-hex-p3'
    if slice_every < 0 or (slice_every and not mapped):
        raise ValueError('periodic diameter slices require the mapped cylindrical jet')
    if fwh_surface is not None and not mapped:
        raise ValueError('FW–H surface sampling requires the mapped cylindrical jet')
    if perturb_at_step and not mapped:
        raise ValueError('one-time shear seeds require the mapped cylindrical jet')
    mesh_config = model.checkpoint_mesh() if mapped else {'edges': [e.tolist() for e in model.s.edges]}
    config = dict(**mesh_config, p=model.s.p, mu=model.mu,
                  mach=model.mach, shear_width=model.shear_width, disturbance=model.disturbance,
                  flux=model.interface_flux, modal_filter=model.modal_filter_name, backend=backend,
                  cfl=cfl, sample_from=sample_from, sample_interval=sample_interval)
    if sample_from_step is not None:
        config['sample_from_step'] = sample_from_step
    if sample_max_steps is not None:
        config['sample_max_steps'] = sample_max_steps
    if equal_weight:
        config['sample_weighting'] = sample_weighting
    if model.outlet != 'characteristic':
        config['outlet'] = model.outlet
    if getattr(model, 'initial_flow', 'rest') != 'rest':
        config['initial_flow'] = model.initial_flow
    first_sample_time = sample_from
    last_sample_time = None
    last_sample_step = None
    advance = model.make_stepper(backend, group_packet=group_packet) if mapped else None
    samples = JetSamples(model, full_field_statistics=full_field_statistics)
    if full_field_statistics:
        config['full_field_statistics'] = True
    started_field_statistics = False
    committed_fwh = None
    saved_perturbation = None
    if resume:
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        saved_perturbation = saved.get('shear_seeds')
        committed_fwh = saved.get('fwh_samples')
        if committed_fwh is not None and fwh_surface is None:
            raise ValueError('resume with --fwh-surface to continue the recorded time series')
        state, t, step = saved['state'], saved['time'], saved['step']
        if final_time < t or (max_steps is not None and max_steps < step):
            raise ValueError('target precedes checkpoint')
        saved_config = saved['config']
        if reset_samples:
            if step > sample_from_step:
                raise ValueError('new sampling start step precedes checkpoint')
            saved_config = dict(saved_config, sample_from=sample_from, sample_interval=sample_interval,
                                sample_from_step=sample_from_step)
            saved_config.pop('sample_max_steps', None)
            saved_config.pop('sample_weighting', None)
            if sample_max_steps is not None:
                saved_config['sample_max_steps'] = sample_max_steps
            if equal_weight:
                saved_config['sample_weighting'] = sample_weighting
        elif not saved['sample_count'] and sample_from is not None and sample_from > t:
            saved_config = dict(saved_config, sample_from=sample_from, sample_interval=sample_interval)
        if full_field_statistics and not saved_config.get('full_field_statistics'):
            if saved['sample_count'] and not reset_samples:
                raise ValueError('full-field statistics must start with a fresh averaging window')
            started_field_statistics = True
            saved_config = dict(saved_config, full_field_statistics=True)
        if saved_config != config:
            raise ValueError('checkpoint mesh, numerical or sampling configuration differs')
        if not reset_samples:
            samples.total.copy_(saved['sample_sum'])
            samples.count = saved['sample_count']
            samples.weight = saved.get('sample_weight', samples.count)
            if samples.field is not None and samples.count:
                samples.field.moments = saved['field_moments']
            if sample_from_step is not None:
                first_sample_time = saved['first_sample_time']
            if sample_max_steps is not None:
                last_sample_time = saved['last_sample_time']
                last_sample_step = saved['last_sample_step']
        del saved
    else:
        if checkpoint.exists():
            raise ValueError('output already has a checkpoint; use --resume or another directory')
        state, t, step = model.initial(), 0., 0
    if advance is None:
        advance = model.make_stepper(backend, group_packet=group_packet)
    seeds = None
    if perturb_at_step or saved_perturbation is not None:
        from .perturbation import ShearSeeds
        seeds = ShearSeeds(model, perturb_at_step, perturb_amplitude, step, saved_perturbation)

    def apply_shear_seed():
        if seeds is None or not seeds.apply(state, step):
            return False
        print(f'one_time_shear_seed step={step} tUj/D={t:.6f} '
              f'max_delta_u/Uj={seeds.amplitude:g}', flush=True)
        return True

    seeded_start = False
    if backend != 'torch':
        state = model.soa(state)
        writable = torch.empty_like(state)
        seeded_start = apply_shear_seed()
        valid, next_dt = advance.scan(state, cfl, t=t)
        if not valid:
            raise RuntimeError('nonphysical starting state')
    if sample_from_step == step and first_sample_time is None:
        first_sample_time = last_sample_time = t
        last_sample_step = step
        if sample_max_steps is None:
            samples.add(state)
    fwh = None
    if fwh_surface is not None:
        from .fwh_sampling import FWHRecorder
        fwh = FWHRecorder(model, output/'fwh', fwh_surface, fwh_interval, t, committed_fwh)
        if committed_fwh is None:
            fwh.record(state,t)
        print(f'FW–H points={fwh.point_count} samples={fwh.count} interval={fwh_interval:g}',
              flush=True)
    next_sample = math.inf
    if first_sample_time is not None and not equal_weight:
        next_sample = (first_sample_time+samples.count*sample_interval if sample_max_steps is None
                       else last_sample_time+sample_interval)

    def save_checkpoint():
        temporary = checkpoint.with_suffix('.tmp')
        payload = dict(config=config, state=state, time=t, step=step,
                       sample_sum=samples.total, sample_count=samples.count,
                       first_sample_time=first_sample_time)
        if sample_max_steps is not None:
            payload.update(sample_weight=samples.weight, last_sample_time=last_sample_time,
                           last_sample_step=last_sample_step)
        if samples.field is not None:
            payload['field_moments'] = samples.field.moments
        if fwh is not None:
            fwh.sync()
            payload['fwh_samples'] = fwh.count
        if seeds is not None:
            payload['shear_seeds'] = seeds.checkpoint()
        torch.save(payload, temporary)
        temporary.replace(checkpoint)

    started = time.perf_counter()
    reported_at, reported_step = started, step
    filter_retries = 0
    if seeded_start or (checkpoint_every and (not resume or reset_samples or started_field_statistics
                                             or (fwh is not None and committed_fwh is None))):
        save_checkpoint()
    if step and slice_every and step % slice_every == 0:
        write_velocity_slice(model, state, output, step)
    while t < final_time and (max_steps is None or step < max_steps):
        if backend == 'torch':
            next_dt = model.timestep(state, cfl, t=t)
        dt = min(next_dt, final_time-t)
        if t < next_sample < t+dt:
            dt = next_sample-t
        if not math.isfinite(dt) or dt <= 0 or t+dt == t:
            raise RuntimeError(f'invalid step size at step {step}, t={t:g}')
        if backend != 'torch':
            _, valid, next_dt = advance.advance_finish(state, t, dt, cfl, out=writable)
            if valid:
                state, writable = writable, state
            elif mapped and 0 < next_dt < dt:
                filter_retries += 1
                continue
        else:
            state = advance(state, t, dt)
            valid = model.eq.valid(state)
        if not valid:
            raise RuntimeError(f'nonphysical state at step {step+1}, t={t+dt:g}')
        previous_time = t
        t += dt
        step += 1
        if fwh is not None:
            fwh.record(state,t,writable,previous_time)
        seeded = apply_shear_seed()
        if seeded:
            valid, next_dt = advance.scan(state,cfl,t=t)
            if not valid:
                raise RuntimeError(f'nonphysical state after shear seed at step {step}')
        if step == sample_from_step:
            first_sample_time = last_sample_time = t
            last_sample_step = step
            if not equal_weight:
                next_sample = t if sample_max_steps is None else t+sample_interval
            print(f'sampling starts at step={step} tUj/D={t:.5f}', flush=True)
        if sample_max_steps is not None:
            if last_sample_step is not None and step > last_sample_step and (
                    t >= next_sample-1e-12 or step-last_sample_step >= sample_max_steps):
                samples.add(state, weight=1. if equal_weight else t-last_sample_time)
                last_sample_time, last_sample_step = t, step
                if not equal_weight:
                    next_sample = t+sample_interval
        elif t >= next_sample-1e-12:
            samples.add(state)
            next_sample = first_sample_time+samples.count*sample_interval
        if seeded or (checkpoint_every and step % checkpoint_every == 0):
            save_checkpoint()
        if slice_every and step % slice_every == 0:
            print(f'slice={write_velocity_slice(model, state, output, step)}', flush=True)
        if step % 100 == 0:
            now = time.perf_counter()
            step_ms = 1000*(now-reported_at)/(step-reported_step)
            retry_report = f' filter_retries={filter_retries}' if filter_retries else ''
            fwh_report = f' fwh_samples={fwh.count}' if fwh is not None else ''
            print(f'step={step} tUj/D={t:.5f} dt={dt:.4g} ms/step={step_ms:.2f}{retry_report}{fwh_report}', flush=True)
            reported_at, reported_step = now, step
            filter_retries = 0
    elapsed = time.perf_counter()-started
    if checkpoint_every or resume:
        save_checkpoint()
    if fwh is not None:
        fwh.close()
    samples.write(state, output, time=t, steps=step, elapsed=elapsed, plot=plot,
                  sample_from=first_sample_time, sample_interval=sample_interval, cfl=cfl,
                  backend=backend, group_packet=group_packet, sample_from_step=sample_from_step,
                  sample_max_steps=sample_max_steps, last_sample_time=last_sample_time,
                  sample_weighting=sample_weighting)
    print(f'{output}: step={step} tUj/D={t:.5f} samples={samples.count} elapsed={elapsed:.2f}s', flush=True)
    return state


def main():
    mesh_options = argparse.ArgumentParser(add_help=False)
    mesh_args = mesh_options.add_mutually_exclusive_group()
    mesh_args.add_argument('--mesh', type=Path)
    mesh_args.add_argument('--cells', type=int, nargs=3)
    mesh_args.add_argument('--hex-mesh', type=Path)
    mesh_options.add_argument('--backend', choices=('native-p3', 'native-hex-p3', 'torch'))
    parser = argparse.ArgumentParser(description=__doc__, parents=[mesh_options])
    parser.add_argument('--output', type=Path, default=Path('results/round_jet_les/near_field'))
    parser.add_argument('--p', type=int, default=3)
    parser.add_argument('--group-packet', type=int, help='native groups per dynamic packet (default 2, 0 for static)')
    parser.add_argument('--reynolds', type=float, default=6e5)
    parser.add_argument('--mach', type=float, default=.5)
    parser.add_argument('--shear-width', type=float, default=.12)
    parser.add_argument('--disturbance', type=float, default=.03)
    parser.add_argument('--flux', choices=('roe', 'rusanov'), default='rusanov')
    parser.add_argument('--modal-filter', choices=('none', 'hifiles'), default='hifiles')
    parser.add_argument('--outlet', choices=('characteristic', 'nscbc', 'nscbc-3d',
                                             'nscbc-3d-open', 'nscbc-3d-unified', 'nscbc-3d-ambient'),
                        default='characteristic')
    parser.add_argument('--time', type=float, default=12.)
    parser.add_argument('--cfl', type=float, default=1.6)
    parser.add_argument('--threads', type=int, default=8)
    sampling_start = parser.add_mutually_exclusive_group()
    sampling_start.add_argument('--sample-from', type=float)
    sampling_start.add_argument('--sample-from-step', type=int)
    parser.add_argument('--sample-interval', type=float, default=.25)
    parser.add_argument('--sample-max-steps', type=int)
    parser.add_argument('--sample-weighting', choices=('time', 'step'), default='time',
                        help='use equal weights with --sample-max-steps 1')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--slice-every', type=int, default=0)
    parser.add_argument('--fwh-surface', type=Path, help='JSON geometry plan for FW–H sampling')
    parser.add_argument('--fwh-interval', type=float, default=.04, help='FW–H time spacing in D/Uj')
    parser.add_argument('--perturb-at-step', type=int, nargs='+', default=(),
                        help='accepted steps at which to apply a one-time transverse shear seed')
    parser.add_argument('--perturb-amplitude', type=float, default=.005,
                        help='peak velocity increment in units of Uj')
    parser.add_argument('--initial-flow', choices=('rest', 'laminar'), default='rest')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reset-samples', action='store_true')
    parser.add_argument('--full-field-statistics', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    selection, _ = mesh_options.parse_known_args()
    configured_mesh = None
    if (not (selection.mesh or selection.cells or selection.hex_mesh)
            and selection.backend not in ('native-p3', 'torch')):
        from .generate_tanh_mesh import load_case_config, write_mesh
        config = load_case_config()
        configured_mesh = config['mesh']
        parser.set_defaults(**config['solver'])
    args = parser.parse_args()
    if configured_mesh is not None:
        args.hex_mesh = args.output/'mesh.msh'
    backend = args.backend or ('native-hex-p3' if args.hex_mesh else 'native-p3')
    sample_from = args.sample_from
    sample_interval = args.sample_interval
    if sample_from is None and args.sample_from_step is None:
        sample_from = args.time/2
    if (not all(math.isfinite(v) and v > 0 for v in (args.time, args.cfl))
            or (sample_from is not None and not 0 < sample_from < args.time)
            or not 0 < sample_interval < math.inf
            or args.threads < 1 or args.checkpoint_every < 0
            or (args.max_steps is not None and args.max_steps < 1)):
        parser.error('require 0 < sample-from < time, positive interval, CFL, threads and optional max steps')
    if args.sample_from_step is not None and (args.sample_from_step < 1
            or (args.max_steps is not None and args.sample_from_step > args.max_steps)):
        parser.error('sample-from-step must be positive and within max steps')
    if args.sample_max_steps is not None and (args.sample_max_steps < 1
                                             or args.sample_from_step is None):
        parser.error('sample-max-steps requires a positive step cap and --sample-from-step')
    if args.sample_weighting == 'step' and args.sample_max_steps != 1:
        parser.error('step weighting requires --sample-max-steps 1')
    if args.reset_samples and (not args.resume or args.sample_from_step is None):
        parser.error('reset-samples requires --resume and --sample-from-step')
    if backend != 'torch' and args.p != 3:
        parser.error('native backends require --p 3')
    if bool(args.hex_mesh) != (backend == 'native-hex-p3'):
        parser.error('native-hex-p3 requires --hex-mesh')
    if args.outlet.startswith('nscbc-3d') and not args.hex_mesh:
        parser.error('3D characteristic boundaries require the native mapped hexahedral backend')
    if (args.slice_every or args.initial_flow != 'rest') and not args.hex_mesh:
        parser.error('periodic slices and laminar initialization require a mapped jet mesh')
    if args.slice_every < 0:
        parser.error('slice interval must be nonnegative')
    if args.hex_mesh and args.flux != 'rusanov':
        parser.error('native-hex-p3 requires Rusanov flux')
    if args.group_packet is not None and (
            not 0 <= args.group_packet <= 2**31-1 or (backend == 'torch' and args.group_packet != 0)):
        parser.error('group-packet must be nonnegative, within int32 range, and zero for Torch')
    torch.set_num_threads(args.threads)
    physical = dict(reynolds=args.reynolds, mach=args.mach, shear_width=args.shear_width,
                    disturbance=args.disturbance, modal_filter=args.modal_filter, outlet=args.outlet)
    if args.hex_mesh:
        from .mapped import MappedRoundJetLES
        if configured_mesh is not None:
            try:
                write_mesh(args.hex_mesh, **configured_mesh, replace=False)
            except (ValueError, FileExistsError) as error:
                parser.error(str(error))
        model = MappedRoundJetLES(args.hex_mesh, initial_flow=args.initial_flow, **physical)
    else:
        mesh = CartesianMesh.read(args.mesh) if args.mesh else build_mesh(tuple(args.cells or (32, 24, 24)))
        model = RoundJetLES(mesh, args.p, interface_flux=args.flux, **physical)
    run(model, args.output, final_time=args.time, cfl=args.cfl, sample_from=sample_from,
        sample_interval=sample_interval, max_steps=args.max_steps, plot=not args.no_plot,
        backend=backend, group_packet=args.group_packet,
        checkpoint_every=args.checkpoint_every, resume=args.resume,
        sample_from_step=args.sample_from_step, reset_samples=args.reset_samples,
        full_field_statistics=args.full_field_statistics, sample_max_steps=args.sample_max_steps,
        sample_weighting=args.sample_weighting, slice_every=args.slice_every,
        fwh_surface=args.fwh_surface, fwh_interval=args.fwh_interval,
        perturb_at_step=args.perturb_at_step, perturb_amplitude=args.perturb_amplitude)
