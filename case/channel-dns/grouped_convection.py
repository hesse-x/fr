"""AVX2 p=3 convection: pack a group, complete convection, scatter.

The persistent state keeps Cartesian SoA layout. Only thread-local scratch is
packed. No global face states, volume fluxes or directional derivatives exist.
"""
import ctypes
import hashlib
import itertools
from pathlib import Path
import subprocess

import torch
from native_build import header_bytes, compiler_flags, locality_options

_LIBRARY = None


def load_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = Path(__file__).with_name('native')/'grouped_convection.cpp'
    flags = compiler_flags()
    key = hashlib.sha256(source.read_bytes()+header_bytes()+' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-channel-native')/key
    build.mkdir(parents=True, exist_ok=True)
    target = build/'grouped_convection.so'
    if not target.exists():
        command = ['g++', *flags, f'-fopt-info-vec-optimized={build / "vectorization.txt"}',
                   str(source), '-o', str(target)]
        (build/'command.txt').write_text(' '.join(command)+'\n')
        result = subprocess.run(command, capture_output=True, text=True)
        (build/'build.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
    library = ctypes.CDLL(str(target))
    p = ctypes.c_void_p
    library.grouped_convection.argtypes = [p, p, ctypes.c_int64]+[p]*12+[
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, p]
    library.grouped_convection.restype = None
    library.build_directory = str(build)
    _LIBRARY = library
    return library


class GroupedConvection:
    shared_fields = False

    def __init__(self, model, block=(4, 2, 4)):
        if model.s.p != 3 or model.s.z.device.type != 'cpu':
            raise ValueError('experimental grouped kernel supports CPU p=3 only')
        if getattr(model.convection, 'flux_mode', None) != 'local':
            raise ValueError('grouped prototype requires the flux-local convection backend')
        if len(block) != 3 or any(n < 1 for n in block):
            raise ValueError('three positive tile dimensions required')
        self.shape = tuple(model.s.shape)+(5,)
        self.block = tuple(block)
        self.cells = tuple(model.s.cells)
        nx, ny, nz = self.cells
        self.count = nx*ny*nz
        self.gamma = model.eq.gamma
        self.library = load_library()
        self.group_offsets, self.face_offsets = [0], [0]
        owned, faces, cell_faces = [], [], []
        for corner in itertools.product(*(range(0, n, b) for n, b in zip(self.cells, block))):
            coords = list(itertools.product(*(range(a, min(a+b, n))
                          for a, b, n in zip(corner, block, self.cells))))
            old_ids = [(x*ny+y)*nz+z for x, y, z in coords]
            local = {old: new for new, old in enumerate(old_ids)}
            face_ids, group_faces = {}, []
            for old, coord in zip(old_ids, coords):
                links = []
                for axis in range(3):
                    for side in (-1, 1):
                        neighbor = list(coord)
                        neighbor[axis] += side
                        wall = 0
                        if not 0 <= neighbor[axis] < self.cells[axis]:
                            if model.s.periodic[axis]:
                                neighbor[axis] %= self.cells[axis]
                            else:
                                wall = side
                        adjacent = -1 if wall else (neighbor[0]*ny+neighbor[1])*nz+neighbor[2]
                        minus, plus = (adjacent, old) if side < 0 else (old, adjacent)
                        key = (axis, minus, plus, wall)
                        if key not in face_ids:
                            face_ids[key] = len(group_faces)
                            group_faces.append((axis, minus, plus,
                                                local.get(minus, -1), local.get(plus, -1), wall))
                        links.append(face_ids[key])
                cell_faces.append(links)
            owned.extend(old_ids)
            faces.extend(group_faces)
            self.group_offsets.append(len(owned))
            self.face_offsets.append(len(faces))
        group_sizes = [b-a for a, b in zip(self.group_offsets, self.group_offsets[1:])]
        face_sizes = [b-a for a, b in zip(self.face_offsets, self.face_offsets[1:])]
        self.groups, self.max_cells, self.max_faces = len(group_sizes), max(group_sizes), max(face_sizes)
        widths = torch.stack(torch.meshgrid(*model.s.widths, indexing='ij'), -1).reshape(-1, 3).contiguous()
        reference = torch.stack(model.convection.reference_flux).contiguous()
        matrix = (model.s.D-model.s.gl[:, None]*model.s.L[None, :]
                  -model.s.gr[:, None]*model.s.R[None, :]).contiguous()
        integer = lambda values: torch.tensor(values, dtype=torch.int64, device='cpu')
        self.buffers = [integer(self.group_offsets), integer(owned),
                        integer(self.face_offsets), integer(faces), integer(cell_faces),
                        widths, model.s.L.contiguous(), model.s.R.contiguous(), matrix,
                        model.s.gl.contiguous(), model.s.gr.contiguous(), reference]
        face_state_slots = 10*16 if locality_options()['face_local'] else 10*self.max_faces*16
        self.scratch_bytes_per_worker = 8*(19*self.max_cells*64+face_state_slots+5*self.max_faces*16)
        self.metadata = dict(block=list(block), groups=self.groups,
                             max_cells=self.max_cells, max_interfaces=self.max_faces,
                             scratch_bytes_per_worker=self.scratch_bytes_per_worker,
                             packed_interfaces=len(faces), build_directory=self.library.build_directory)

    def __call__(self, state, t=0.):
        return self.evaluate(state)

    def evaluate(self, state, *, profile=False):
        if (state.device.type != 'cpu' or state.dtype != torch.float64 or
                tuple(state.shape) != self.shape or not state.movedim(-1, 0).is_contiguous()):
            raise ValueError('expected Cartesian CPU float64 SoA state matching the plan')
        output = torch.empty_strided(state.shape, state.stride(), dtype=state.dtype, device=state.device)
        timing = (ctypes.c_double*4)() if profile else None
        self.library.grouped_convection(state.data_ptr(), output.data_ptr(), self.count,
            *(buffer.data_ptr() for buffer in self.buffers), self.gamma,
            self.groups, self.max_cells, self.max_faces, torch.get_num_threads(), timing)
        if profile:
            return output, dict(zip(('pack_and_face_traces', 'numerical_flux',
                                     'primitive_volume_and_lifting', 'scatter'), timing))
        return output
