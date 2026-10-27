"""Group-local convection plus LDG without a full-grid convective residual."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

import torch

from grouped_convection import GroupedConvection
from grouped_ldg import GroupedLDG
from native_build import header_bytes, compiler_flags, locality_options

_LIBRARIES = {}


def load_library():
    source = Path(__file__).with_name('native')/'grouped_fused.cpp'
    flags = compiler_flags()
    key = hashlib.sha256(source.read_bytes()+header_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    if key in _LIBRARIES:
        return _LIBRARIES[key]
    build = Path('/tmp/fr-channel-native')/key
    build.mkdir(parents=True, exist_ok=True)
    target = build/'grouped_fused.so'
    if not target.exists():
        command = ['g++', *flags, str(source), '-o', str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        (build/'command.txt').write_text(' '.join(command)+'\n')
        (build/'build.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    lib = ctypes.CDLL(str(target))
    p, i, n = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64
    lib.grouped_fused.argtypes = [p, p, n, p, p, p, i, i, i, i, i, i, i, p]
    lib.grouped_fused.restype = None
    lib.grouped_fused_rk.argtypes = [p, p, n, p, p, p, i, i, i, i, i, i, i, p, p, ctypes.c_double, i]
    lib.grouped_fused_rk.restype = None
    lib.build_directory = str(build)
    _LIBRARIES[key] = lib
    return lib


class GroupedFused:
    def __init__(self, model, block=(4, 4, 4)):
        self.convection = GroupedConvection(model, block)
        self.ldg = GroupedLDG(model, block)
        conv, ldg = self.convection, self.ldg
        if conv.groups != ldg.groups:
            raise ValueError('mismatched group partitions')
        offsets = ldg.buffers[0].tolist()
        for g in range(conv.groups):
            start, end = conv.group_offsets[g:g+2]
            bs = offsets[g][2]
            if not torch.equal(conv.buffers[1][start:end], ldg.buffers[1][bs:bs+end-start]):
                raise ValueError('mismatched owned-cell order')
        self.library = load_library()
        self.shape = tuple(model.s.shape)+(5,)
        self.count = ldg.count
        self.conv_pointers = (ctypes.c_void_p*len(conv.buffers))(*(v.data_ptr() for v in conv.buffers))
        self.ldg_pointers = (ctypes.c_void_p*len(ldg.buffers))(*(v.data_ptr() for v in ldg.buffers))
        self.constants = (ctypes.c_double*5)(*ldg.constants)
        self.metadata = dict(block=list(block), groups=conv.groups,
                             dependency_pruned_halo=True,
                             scratch_bytes_per_worker=conv.scratch_bytes_per_worker+
                             ldg.metadata['scratch_bytes_per_worker'],
                             global_convection_buffer_bytes_avoided=self.count*64*5*8,
                             primitive_reuse=False,
                             **locality_options(),
                             build_directory=self.library.build_directory)

    def __call__(self, state, t=0.):
        return self.evaluate(state)

    def evaluate(self, state, *, profile=False):
        if (state.device.type != 'cpu' or state.dtype != torch.float64 or tuple(state.shape) != self.shape
                or not state.movedim(-1, 0).is_contiguous()):
            raise ValueError('expected matching CPU float64 SoA state')
        output = torch.empty_strided(state.shape, state.stride(), dtype=state.dtype)
        timing = (ctypes.c_double*8)() if profile else None
        conv, ldg = self.convection, self.ldg
        self.library.grouped_fused(state.data_ptr(), output.data_ptr(), self.count,
            self.conv_pointers, self.ldg_pointers, self.constants, conv.groups, conv.max_cells,
            conv.max_faces, ldg.max_owned, ldg.max_active, ldg.max_primitive, torch.get_num_threads(), timing)
        return (output, list(timing)) if profile else output

    def make_rhs(self):
        return lambda state, t=0.: self.evaluate(state)

    def rk_stage(self, original, stage, output, pending, coefficient, combined):
        if self.ldg.constants[-1] != 0:
            raise ValueError('local constant-flux RK requires an unforced plan')
        for tensor, components in ((original, 5), (stage, 5), (output, 5), (pending, 2)):
            if (tensor.device.type != 'cpu' or tensor.dtype != torch.float64
                    or tuple(tensor.shape) != self.shape[:-1]+(components,)
                    or not tensor.movedim(-1, 0).is_contiguous()):
                raise ValueError('expected matching SoA RK fields')
        if self.metadata['local_streaming_stores'] and output.data_ptr() % 32:
            raise ValueError('streaming RK output must be 32-byte aligned')
        conv, ldg = self.convection, self.ldg
        self.library.grouped_fused_rk(stage.data_ptr(), output.data_ptr(), self.count,
            self.conv_pointers, self.ldg_pointers, self.constants, conv.groups, conv.max_cells,
            conv.max_faces, ldg.max_owned, ldg.max_active, ldg.max_primitive, torch.get_num_threads(),
            original.data_ptr(), pending.data_ptr(), coefficient, int(combined))
