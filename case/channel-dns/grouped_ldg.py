"""AVX2 p=3 LDG: private halo gradients, stresses and dual lifting."""
import ctypes
import hashlib
import itertools
from pathlib import Path
import subprocess

import torch
from native_build import header_bytes, compiler_flags

_LIBRARY = None


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_name('native')/'grouped_ldg.cpp'
    flags = compiler_flags()
    key = hashlib.sha256(source.read_bytes()+header_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-channel-native')/key
    build.mkdir(parents=True, exist_ok=True)
    target = build/'grouped_ldg.so'
    if not target.exists():
        command = ['g++', *flags, f'-fopt-info-vec-optimized={build / "vectorization.txt"}',
                   str(source), '-o', str(target)]
        (build/'command.txt').write_text(' '.join(command)+'\n')
        result = subprocess.run(command, capture_output=True, text=True)
        (build/'build.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    library = ctypes.CDLL(str(target))
    p = ctypes.c_void_p
    library.grouped_ldg.argtypes = [p, p, p, ctypes.c_int64]+[p]*11+[
        ctypes.c_double]*5+[ctypes.c_int]*5+[p]
    library.grouped_ldg.restype = None
    library.build_directory = str(build)
    _LIBRARY = library
    return library


class GroupedLDG:
    def __init__(self, model, block=(4, 2, 4)):
        if (model.s.p != 3 or model.s.z.device.type != 'cpu'
                or model.s.periodic != (True, False, True)):
            raise ValueError('expected CPU p=3 directional fused channel LDG')
        if len(block) != 3 or any(n < 1 for n in block):
            raise ValueError('three positive tile dimensions required')
        self.shape = tuple(model.s.shape)+(5,)
        self.cells = tuple(model.s.cells)
        nx, ny, nz = self.cells
        self.count = nx*ny*nz
        self.constants = (model.eq.gamma, model.mu, model.kappa, model.temperature, model.forcing)
        self.library = load_library()

        def coord(old):
            return [old//(ny*nz), old//nz % ny, old % nz]

        def neighbor(old, axis, step):
            c = coord(old)
            c[axis] += step
            if not 0 <= c[axis] < self.cells[axis]:
                if not model.s.periodic[axis]:
                    return -1
                c[axis] %= self.cells[axis]
            return (c[0]*ny+c[1])*nz+c[2]

        offsets, primitive_ids, active_meta, owned_meta = [[0, 0, 0]], [], [], []
        sizes = []
        for corner in itertools.product(*(range(0, n, b) for n, b in zip(self.cells, block))):
            coords = itertools.product(*(range(a, min(a+b, n))
                                        for a, b, n in zip(corner, block, self.cells)))
            owned = [(x*ny+y)*nz+z for x, y, z in coords]
            active = list(owned)
            active_index = {old: i for i, old in enumerate(active)}
            for old in owned:
                for axis in range(3):
                    other = neighbor(old, axis, 1)
                    if other >= 0 and other not in active_index:
                        active_index[other] = len(active)
                        active.append(other)
            packed = list(active)
            packed_index = {old: i for i, old in enumerate(packed)}
            for old in active:
                for axis in range(3):
                    other = neighbor(old, axis, -1)
                    if other >= 0 and other not in packed_index:
                        packed_index[other] = len(packed)
                        packed.append(other)
            for old in active:
                minus = [packed_index.get(neighbor(old, axis, -1), -1) for axis in range(3)]
                y = old//nz % ny
                active_meta.append([*minus, int(y == 0), int(y == ny-1)])
            for old in owned:
                owned_meta.append([active_index.get(neighbor(old, axis, 1), -1) for axis in range(3)])
            primitive_ids.extend(packed)
            offsets.append([len(owned_meta), len(active_meta), len(primitive_ids)])
            sizes.append((len(owned), len(active), len(packed)))
        self.groups = len(sizes)
        self.max_owned, self.max_active, self.max_primitive = (max(s[i] for s in sizes) for i in range(3))
        integer = lambda values: torch.tensor(values, dtype=torch.int64, device='cpu')
        widths = torch.stack(torch.meshgrid(*model.s.widths, indexing='ij'), -1).reshape(-1, 3).contiguous()
        self.buffers = [integer(offsets), integer(primitive_ids), integer(active_meta), integer(owned_meta),
                        widths, model.s.L.contiguous(), model.s.R.contiguous(),
                        (model.s.D-model.s.gl[:, None]*model.s.L[None, :]).contiguous(),
                        (model.s.D-model.s.gr[:, None]*model.s.R[None, :]).contiguous(),
                        model.s.gl.contiguous(), model.s.gr.contiguous()]
        self.metadata = dict(block=list(block), groups=self.groups,
            dependency_pruned_halo=True,
            max_owned=self.max_owned, max_active=self.max_active, max_primitive=self.max_primitive,
            total_owned=len(owned_meta), total_active=len(active_meta), total_primitive=len(primitive_ids),
            gradient_halo_ratio=len(active_meta)/self.count,
            primitive_halo_ratio=len(primitive_ids)/self.count,
            scratch_bytes_per_worker=8*64*(4*self.max_primitive+16*self.max_active+5*self.max_owned),
            build_directory=self.library.build_directory)

    def __call__(self, state, convection):
        return self.evaluate(state, convection)

    def evaluate(self, state, convection, *, profile=False):
        for tensor in (state, convection):
            if (tensor.device.type != 'cpu' or tensor.dtype != torch.float64 or
                    tuple(tensor.shape) != self.shape or not tensor.movedim(-1, 0).is_contiguous()):
                raise ValueError('expected matching Cartesian CPU float64 SoA tensors')
        output = torch.empty_strided(state.shape, state.stride(), dtype=state.dtype, device=state.device)
        timing = (ctypes.c_double*5)() if profile else None
        self.library.grouped_ldg(state.data_ptr(), convection.data_ptr(), output.data_ptr(), self.count,
            *(buffer.data_ptr() for buffer in self.buffers), *self.constants,
            self.groups, self.max_owned, self.max_active, self.max_primitive, torch.get_num_threads(), timing)
        if profile:
            return output, dict(zip(('pack_primitive', 'gradient', 'stress_and_heat', 'dual_lifting', 'wall_and_scatter'), timing))
        return output

    def make_rhs(self, convection):
        return lambda state, t=0.: self.evaluate(state, convection(state))
