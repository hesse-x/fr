"""Cached C ABI library, with persistent native scratch and fused RK4 algebra."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

import torch
from fr.native import header_bytes

_LIBRARY = None


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_suffix('.cpp')
    flags = ['-O3', '-std=c++17', '-march=native', '-fopenmp',
             '-ffp-contract=fast', '-fno-math-errno', '-shared', '-fPIC']
    key = hashlib.sha256(source.read_bytes()+header_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-parabolic-native')/key
    build.mkdir(parents=True, exist_ok=True)
    target = build/'antenna.so'
    if not target.exists():
        # A separate temporary target avoids loading an incomplete shared object.
        import os
        temporary = build/f'antenna-{os.getpid()}.so'
        command = ['g++', *flags, str(source), '-o', str(temporary)]
        result = subprocess.run(command, capture_output=True, text=True)
        (build/'command.txt').write_text(' '.join(command)+'\n')
        (build/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'native compilation failed: {result.stderr}')
        temporary.replace(target)
    lib = ctypes.CDLL(str(target))
    p, i, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    lib.antenna_create.argtypes = [i, i, i, i, i, p, p, p, p, p, i, p, p, p, i, d]
    lib.antenna_create.restype = p
    lib.antenna_voltage.argtypes = [p, p]
    lib.antenna_voltage.restype = d
    lib.antenna_destroy.argtypes = [p]
    lib.antenna_destroy.restype = None
    lib.antenna_rhs.argtypes = [p, p, p, d]
    lib.antenna_rhs.restype = None
    lib.antenna_step.argtypes = [p, p, p, d, d]
    lib.antenna_step.restype = None
    _LIBRARY = lib
    return lib


class NativeMaxwell:
    """CPU FP64; each instance owns its scratch and is used sequentially."""
    def __init__(self, case):
        self.shape = case.initial.shape
        self.library = load_library()
        operators = torch.cat([v.flatten() for v in (case.D, case.L, case.R, case.gl, case.gr)]).contiguous()
        widths = torch.broadcast_tensors(case.widths[0][:, None, None], case.widths[1][None, :, None], case.widths[2][None, None, :])
        h = torch.stack(widths, -1).contiguous()
        rates = case.damping.squeeze(-1).contiguous()
        drive = case.drive_cos.contiguous()
        solid = case.solid.to(torch.uint8).contiguous()
        self.handle = self.library.antenna_create(*case.cells, case.p+1, torch.get_num_threads(),
            h.data_ptr(), operators.data_ptr(), rates.data_ptr(), drive.data_ptr(), solid.data_ptr(),
            int(case.outer_boundary == 'silver-muller'), case.feed_ids.data_ptr(),
            case.feed_coeff.data_ptr(), case.feed_lift.data_ptr(), len(case.feed_ids), case.voltage_amplitude)
        if not self.handle:
            raise RuntimeError('native solver allocation failed')

    def __del__(self):
        if getattr(self, 'handle', None):
            self.library.antenna_destroy(self.handle)

    def check(self, state):
        if state.device.type != 'cpu' or state.dtype != torch.float64 or state.shape != self.shape or not state.is_contiguous():
            raise ValueError('native Maxwell requires matching contiguous CPU FP64 state')

    def voltage(self, state):
        self.check(state)
        return self.library.antenna_voltage(self.handle, state.data_ptr())

    def rhs(self, state, t):
        self.check(state)
        result = torch.empty_like(state)
        self.library.antenna_rhs(self.handle, state.data_ptr(), result.data_ptr(), t)
        return result

    def step(self, rhs, state, t, dt):
        self.check(state)
        result = torch.empty_like(state)
        self.library.antenna_step(self.handle, state.data_ptr(), result.data_ptr(), t, dt)
        return result
