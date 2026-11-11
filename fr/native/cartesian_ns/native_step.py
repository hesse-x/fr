"""Fused native constant-flux RK updates and combined validity/CFL scan."""
import ctypes
import hashlib
import math
from pathlib import Path
import subprocess

import torch
from fr.time import ssprk43_stages

from .native_build import header_bytes, compiler_flags
from .validation import validate_model

_LIBRARY = None


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_name('kernels')/'step_ops.cpp'
    flags = compiler_flags()
    key = hashlib.sha256(source.read_bytes()+header_bytes()+torch.__version__.encode()+' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-cartesian-ns-native')/key
    build.mkdir(parents=True, exist_ok=True)
    target = build/'step_ops.so'
    if not target.exists():
        command = ['g++', *flags, str(source), '-o', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        (build/'command.txt').write_text(' '.join(command)+'\n')
        (build/'build.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    lib = ctypes.CDLL(str(target))
    p, i, d, n = ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int64
    lib.flux_force.argtypes = [p, p, p, n]
    lib.flux_force.restype = d
    lib.flux_rk_update.argtypes = [p, p, p, p, n, d, d, i]
    lib.flux_rk_update.restype = None
    lib.flux_rk_finalize.argtypes = lib.flux_rk_update.argtypes
    lib.flux_rk_finalize.restype = None
    lib.state_scan.argtypes = [p, p, n, d, p]
    lib.state_scan.restype = None
    lib.mean_accumulate.argtypes = [p, p, p, n, d, d]
    lib.mean_accumulate.restype = None
    lib.flux_rk_finish.argtypes = [p, p, p, p, p, p, n, d, d, i, d, d, p]
    lib.flux_rk_finish.restype = None
    _LIBRARY = lib
    return lib


class NativeMeanAccumulator:
    """Trapezoidal time moments directly from two SoA conservative states."""
    def __init__(self, model):
        validate_model(model)
        if model.s.p != 3 or model.s.z.device.type != 'cpu':
            raise ValueError('expected CPU p=3 Cartesian NS')
        self.shape = tuple(model.s.shape)
        self.count = math.prod(self.shape)
        self.gamma = model.eq.gamma
        self.library = load_library()

    def zeros(self):
        return torch.zeros((12, *self.shape), dtype=torch.float64).movedim(0, -1)

    def add(self, integral, previous, current, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite sampling duration required')
        for tensor, components in ((previous, 5), (current, 5), (integral, 12)):
            if (tensor.device.type != 'cpu' or tensor.dtype != torch.float64
                    or tuple(tensor.shape) != self.shape+(components,)
                    or not tensor.movedim(-1, 0).is_contiguous()):
                raise ValueError('expected matching CPU float64 SoA fields')
        self.library.mean_accumulate(previous.data_ptr(), current.data_ptr(), integral.data_ptr(),
                                     self.count, self.gamma, .5*dt)


class NativeFluxStepper:
    def __init__(self, model, base_rhs, *, constant_mass_flux=True):
        validate_model(model)
        if (constant_mass_flux and model.forcing != 0) or model.s.p != 3 or model.s.z.device.type != 'cpu':
            raise ValueError('expected CPU p=3 Cartesian NS; constant flux needs forcing=0')
        self.model, self.base_rhs = model, base_rhs
        self.constant_mass_flux = bool(constant_mass_flux)
        self.library = load_library()
        self.cells = model.s.cells
        self.elements = self.cells[0]*self.cells[1]*self.cells[2]
        self.count = self.elements*64
        widths = torch.broadcast_tensors(model.s.widths[0][:, None, None],
                                         model.s.widths[1][None, :, None],
                                         model.s.widths[2][None, None, :])
        self.widths = torch.stack(widths, -1).reshape(-1, 3).contiguous()
        self.scales = (self.widths.prod(-1)/8/model.volume).contiguous()
        self.weights = model.s.weights.contiguous()
        self.diffusion_factor = (model.s.p+1)**4*sum(float(h.min())**-2 for h in model.s.widths)*max(
            4/3*model.mu, model.kappa*(model.eq.gamma-1))
        self.last_force = None

    def __call__(self, state, t, dt):
        return self.advance(state, t, dt)

    def _force(self, rhs):
        if not self.constant_mass_flux:
            return 0.
        return self.library.flux_force(rhs.data_ptr(), self.weights.data_ptr(),
                                       self.scales.data_ptr(), self.elements)

    @staticmethod
    def _check_time(t, dt):
        if not math.isfinite(t) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite time and positive finite dt required')

    def check(self, state):
        if (state.dtype != torch.float64 or state.device.type != 'cpu'
                or tuple(state.shape) != self.cells+(4, 4, 4, 5)
                or not state.movedim(-1, 0).is_contiguous()):
            raise ValueError('expected matching CPU float64 SoA state')

    def scan(self, state, cfl):
        if not math.isfinite(cfl) or cfl <= 0:
            raise ValueError('positive finite CFL required')
        self.check(state)
        result = (ctypes.c_double*3)()
        self.library.state_scan(state.data_ptr(), self.widths.data_ptr(), self.elements,
                                self.model.eq.gamma, result)
        valid = bool(result[2])
        if not valid:
            return False, float('nan')
        dt = min(cfl/((2*self.model.s.p+1)*result[1]), cfl*result[0]/self.diffusion_factor)
        return True, dt

    def advance(self, state, t, dt):
        self._check_time(t, dt)
        self.check(state)
        original, stage = state, state
        # Native source application and RK algebra share one output pass.
        for stage_time, coefficient, combined in ssprk43_stages(t, dt):
            rhs = self.base_rhs(stage, stage_time)
            self.last_force = self._force(rhs)
            output = torch.empty_strided(stage.shape, stage.stride(), dtype=stage.dtype)
            arguments = (original.data_ptr(), stage.data_ptr(), rhs.data_ptr(), output.data_ptr(),
                         self.count, self.last_force, coefficient, int(combined))
            self.library.flux_rk_update(*arguments)
            stage = output
        return stage


class NativeLocalRKStepper(NativeFluxStepper):
    """Finish three RK components inside the RHS group; defer two until G is known.

    Pending stores [energy RHS, streamwise momentum RHS], so the original
    deterministic flux_force reduction and arithmetic order are unchanged.
    Global stage barriers and all intercell dependencies are retained.
    """
    def __init__(self, model, plan, *, constant_mass_flux=True):
        if plan.model is not model:
            raise ValueError('native plan and stepper must share the same NS model')
        super().__init__(model, None, constant_mass_flux=constant_mass_flux)
        self.plan = plan
        self.pending = torch.empty((2, *model.s.shape), dtype=torch.float64).movedim(0, -1)

    def advance(self, state, t, dt):
        self._check_time(t, dt)
        self.check(state)
        original, stage = state, state
        for _, coefficient, combined in ssprk43_stages(t, dt):
            output = torch.empty_strided(stage.shape, stage.stride(), dtype=stage.dtype)
            self.plan.rk_stage(original, stage, output, self.pending, coefficient, combined)
            self.last_force = self._force(self.pending)
            self.library.flux_rk_finalize(original.data_ptr(), stage.data_ptr(), self.pending.data_ptr(),
                output.data_ptr(), self.count, self.last_force, coefficient, int(combined))
            stage = output
        return stage

    def advance_finish(self, state, t, dt, cfl, integral=None):
        """Fuse final source update, validity, next CFL and optional moments.

        On invalid return discard the step: an in-place integral may be partly
        updated. The production caller raises before committing any checkpoint.
        """
        self._check_time(t, dt)
        if not math.isfinite(cfl) or cfl <= 0:
            raise ValueError('positive finite CFL required')
        self.check(state)
        if integral is not None and (integral.device.type != 'cpu' or integral.dtype != torch.float64
                or tuple(integral.shape) != self.model.s.shape+(12,)
                or not integral.movedim(-1, 0).is_contiguous()):
            raise ValueError('expected matching SoA time integral')
        original, stage = state, state
        stages = ssprk43_stages(t, dt)
        for index, (_, coefficient, combined) in enumerate(stages):
            output = torch.empty_strided(stage.shape, stage.stride(), dtype=stage.dtype)
            self.plan.rk_stage(original, stage, output, self.pending, coefficient, combined)
            self.last_force = self._force(self.pending)
            if index == 3:
                result = (ctypes.c_double*3)()
                self.library.flux_rk_finish(original.data_ptr(), stage.data_ptr(), self.pending.data_ptr(),
                    output.data_ptr(), self.widths.data_ptr(), integral.data_ptr() if integral is not None else None,
                    self.elements, self.last_force, coefficient, int(combined), self.model.eq.gamma,
                    .5*dt, result)
                valid = bool(result[2])
                next_dt = min(cfl/((2*self.model.s.p+1)*result[1]),
                              cfl*result[0]/self.diffusion_factor) if valid else float('nan')
            else:
                self.library.flux_rk_finalize(original.data_ptr(), stage.data_ptr(), self.pending.data_ptr(),
                    output.data_ptr(), self.count, self.last_force, coefficient, int(combined))
            stage = output
        return stage, valid, next_dt
