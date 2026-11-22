"""Native p=3 FR/LDG, WALE and SSPRK43 for the mapped round jet."""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from ..hex_groups import HexGroups, pack_trace_permutations
from ..modal_filter import ConcentrationModalFilter
from ..operators import operators


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix('.cpp')
    files = [source, *sorted(source.parent.glob('*.hpp')),
             *sorted((source.parent/'cartesian_ns'/'kernels').glob('*.hpp'))]
    flags = ['-std=c++17', '-O3', '-march=native', '-fopenmp', '-ffp-contract=fast',
             '-fno-math-errno', '-DFR_NS_WALE=1', '-shared', '-fPIC']
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files)
                            + ' '.join(flags).encode()).hexdigest()[:16]
    directory = Path('/tmp/fr-mapped-jet-native')/digest
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/'mapped_jet.so'
    if not target.exists():
        temporary = directory/f'mapped_jet.{os.getpid()}.so'
        result = subprocess.run(['g++', *flags, str(source), '-o', str(temporary)],
                                capture_output=True, text=True)
        (directory/'build.log').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        temporary.replace(target)
    lib = ctypes.CDLL(str(target))
    ptr, count, real, integer = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double, ctypes.c_int
    lib.mapped_jet_rhs.argtypes = [count, ptr, ptr, ptr, real, real, real, integer, ptr,
                                   ptr, ptr, real, integer, integer, ptr]
    lib.mapped_jet_rhs.restype = integer
    lib.mapped_hex_stage.argtypes = [count, ptr, ptr, ptr, real, integer, integer]
    lib.mapped_hex_stage.restype = None
    lib.mapped_hex_modal_filter.argtypes = [count, ptr, ptr, ptr, ptr, ptr, integer, real, integer,
                                             ptr, ptr, ptr]
    lib.mapped_hex_modal_filter.restype = integer
    return lib


def _pointers(arrays):
    return (ctypes.c_void_p*len(arrays))(*(
        a.data_ptr() if torch.is_tensor(a) else a.ctypes.data for a in arrays))


class MappedJetStepper:
    """One process owns the stage buffers and a spatially grouped native plan."""

    def __init__(self, mesh, block_shapes, *, eq, pressure, shear_width, disturbance,
                 nozzle_radius, outlet, threads, group_packet, modal_filter,
                 split_geometry=False):
        outlet_mode = {'characteristic': 0, 'nscbc': 1, 'nscbc-3d': 2, 'nscbc-3d-open': 3}[outlet]
        self.eq, self.threads = eq, threads
        self.lib = _library()
        ne = mesh.nelements
        self.shape = (ne, 4, 4, 4, 5)
        z, w, D, L, R, gl, gr = operators(3, 'cpu')
        operators_np = tuple(x.numpy() for x in (D, L, R, gl, gr))
        weights = np.einsum('i,j,k->ijk', w.numpy(), w.numpy(), w.numpy()).ravel()
        original = SimpleNamespace(ne=ne, cells=np.arange(ne),
                                   mesh=SimpleNamespace(blocks=[
                                       SimpleNamespace(shape=tuple(n+1 for n in shape))
                                       for shape in block_shapes]),
                                   J=np.empty((ne, 64)), C=np.empty((ne, 64, 3, 3)),
                                   S=np.empty((ne, 6, 16, 3)))
        tag_for = {'inlet': 4, 'outlet': 5, 'ambient': 3}
        tags = np.zeros((ne, 6), dtype=np.int64)
        for (dim, tag), name in mesh.physical_names.items():
            if dim == 2 and name in tag_for:
                tags[mesh.boundary_tags.numpy() == tag] = tag_for[name]
        other = mesh.neighbor_cell.numpy()
        if not np.array_equal(tags == 0, other >= 0):
            raise ValueError('expected inlet, outlet and ambient tags on all exterior faces')
        exterior = tags != 0
        coordinates = np.empty((np.count_nonzero(exterior), 16, 3))
        boundary_id = np.full((ne, 6), -1)
        boundary_id[exterior] = np.arange(len(coordinates))
        for start in range(0, ne, 1024):
            cells = slice(start, start+1024)
            metrics = mesh.volume_metrics(cells=cells)
            original.J[cells] = metrics.jacobian.reshape(-1, 64).numpy()
            original.C[cells] = metrics.contravariant.reshape(-1, 64, 3, 3).numpy()
            for face in range(6):
                position, vectors = mesh.face_metrics(face, cells=cells)
                original.S[cells, face] = vectors.reshape(-1, 16, 3).numpy()
                ids = boundary_id[cells, face]
                coordinates[ids[ids >= 0]] = position[ids >= 0].reshape(-1, 16, 3).numpy()
                axial = tags[cells, face] >= 4
                if np.any(axial):
                    expected = np.where(tags[cells, face][axial] == 4, -1, 1)
                    s = original.S[cells, face][axial]
                    if not np.all(s[..., 0]*expected[:, None] > 0) or np.max(np.abs(s[..., 1:])) > 1e-10:
                        raise ValueError('inlet/outlet faces must align with the physical x axis')
                    if outlet_mode == 1 and face % 2 == 0 and np.any(expected > 0):
                        raise ValueError('the acoustic outlet needs a positive reference face')
        if outlet_mode >= 2:
            open_faces = (tags == 5) | ((outlet_mode == 3) & (tags == 3))
            outlet_metrics = np.zeros((len(coordinates), 16, 13))
            for face in range(6):
                cells = np.flatnonzero(open_faces[:, face])
                if not len(cells):
                    continue
                metric = mesh.face_geometry(face, cells=torch.from_numpy(cells))
                jacobian = metric.jacobian.reshape(-1, 16).numpy()
                cofactor = metric.contravariant.reshape(-1, 16, 3, 3).numpy()
                ids = boundary_id[cells, face]
                outlet_metrics[ids, :, :9] = (cofactor/jacobian[..., None, None]).reshape(-1, 16, 9)
                axis = face//2
                volume_j = np.moveaxis(original.J[cells].reshape(-1, 4, 4, 4), axis+1, 1)
                weighted_lift = R*gr if face % 2 else -L*gl
                surface = original.S[cells, face]
                area = np.linalg.norm(surface, axis=-1)
                trace_lift = np.einsum('i,ei...->e...', weighted_lift.numpy(), 1/volume_j).reshape(-1, 16)*area
                if not np.all(np.isfinite(trace_lift) & (trace_lift > 0)):
                    raise ValueError('the acoustic outlet requires a positive finite face lift')
                outlet_metrics[ids, :, 9] = 1/trace_lift
                outlet_metrics[ids, :, 10:] = surface/area[..., None]
        faces = np.arange(ne*6).reshape(ne, 6)
        target_face = np.where(exterior, faces, other*6+mesh.neighbor_face.numpy())
        transform = mesh.face_permutations(3).numpy()[mesh.face_transform.numpy()]
        original.neighbor_points = target_face[..., None]*16+transform
        original.face_neighbor = target_face
        original.tags = tags
        original.volume = original.J @ weights
        # One lower-side primal trace per shared face, independent of group ownership.
        choose = (original.S.mean(2).sum(2) >= 0).ravel()
        forward = faces.ravel() < target_face.ravel()
        choose[target_face.ravel()[forward]] = ~choose[forward]
        original.primal_inside = choose.reshape(ne, 6).astype(np.int64)
        group = HexGroups(original, shared_faces=True)
        self.order, self.inverse = group.order, group.inverse
        geom = group.geometry
        metric = np.ascontiguousarray(geom.C.transpose(0, 2, 3, 1))
        surface = np.ascontiguousarray(geom.S.transpose(0, 1, 3, 2))
        reordered = boundary_id[group.order][geom.tags != 0]
        yz = coordinates[reordered,:,1:].transpose(0, 2, 1).ravel()
        self.jet = np.r_[pressure, shear_width, disturbance, nozzle_radius,
                         outlet_mode, float(1/(R @ gr)), (R @ D).numpy(), yz]
        self.time = np.zeros(1)
        self.scan_values = np.zeros(2)
        self.timing = np.zeros(3)
        self.jacobian = geom.J
        empty = np.empty(0)
        nb = len(reordered)
        far = np.array([1., 0., 0., 0., pressure/(eq.gamma-1)])
        self.data = [np.reciprocal(geom.J), metric, surface, geom.neighbor_points, geom.tags, geom.primal_inside,
                     *operators_np, far, empty, empty, empty, empty, geom.boundary_index,
                     np.empty((nb,16,5)), self.timing, np.array([1], dtype=np.int64),
                     np.array([.5,.9,1.]), np.cbrt(geom.volume)/4, np.zeros(3), empty, empty,
                     empty, empty, self.jet, self.time]
        if outlet_mode >= 2:
            length = float(mesh.points[:, 0].max()-mesh.points[:, 0].min())
            # Partial transverse compensation limits reflection of oblique waves.
            self.data.extend((np.ascontiguousarray(outlet_metrics[reordered]),
                              np.r_[.25/length, .25, (L @ D).numpy()]))
        # The shared kernel's optional mortar slots must be actual null pointers.
        self.data_ptr = _pointers(self.data)
        self.data_ptr[25] = self.data_ptr[26] = None
        sizes = np.array([len(group.records), group.max_scratch_doubles, 0,
                          int(group_packet > 0), group_packet, 1, 0], dtype=np.int64)
        indices, constants, masks = empty, empty, empty
        self.cartesian_cells = 0
        if split_geometry:
            # Cylinder reference axes are ordered (y, z, x).
            cycle = (1, 2, 0)
            cofactor = geom.C[:, 0, np.arange(3), cycle]
            rect = (cofactor > 0).all(axis=1)
            for a, x in enumerate(cycle):
                rect &= np.isclose(geom.C[:, :, a, x], cofactor[:, a, None],
                                   rtol=2e-12, atol=1e-13).all(axis=1)
                for d in set(range(3))-{x}:
                    rect &= (np.abs(geom.C[:, :, a, d]) <= 1e-13).all(axis=1)
            rect &= np.isclose(geom.J, geom.J[:, :1], rtol=2e-12, atol=1e-13).all(axis=1)
            self.cartesian_cells = int(rect.sum())
            if self.cartesian_cells:
                indices = np.where(rect, -1, np.arange(ne, dtype=np.int64))
                constants = np.zeros((ne, 8))
                constants[rect, 1] = 1/geom.J[rect, 0]
                constants[rect, 2:5] = cofactor[rect]
                constants[rect, 5:] = cofactor[rect]*constants[rect, 1:2]
                masks = np.full(ne, 511, dtype=np.int64)
        self.plan = [*group.arrays, np.empty((threads, group.max_scratch_doubles)),
                     np.empty((nb,5,16)), np.empty((nb,6,16)), empty, empty, sizes,
                     group.shared_sources, group.owned_offsets, group.shared_peers, indices, constants,
                     pack_trace_permutations(geom.neighbor_points), masks]
        self.plan_ptr = _pointers(self.plan)
        if not self.cartesian_cells:
            self.plan_ptr[17] = None
        self.metadata = group.metadata
        self._rhs_buffer = self._stage_a = self._stage_b = None
        self._scanned = None
        self.filter = None
        if modal_filter:
            closure = ConcentrationModalFilter(.1, 36., 2)
            matrix, sensor = closure.matrices(SimpleNamespace(z=z, D=D, p=3))
            self.filter = weights, matrix, sensor, np.zeros(4), np.zeros(ne), np.zeros(ne,dtype=np.int64)

    def _evaluate(self, state, t, *, original=None, target=None, dt=0., stage=-1,
                  scan_stage=False):
        self.time[0] = t
        status = self.lib.mapped_jet_rhs(
            self.shape[0], state.data_ptr(), self._rhs_buffer.data_ptr(), self.data_ptr,
            self.eq.gamma, self.eq.mu, self.eq.prandtl, self.threads, self.scan_values.ctypes.data,
            None if original is None else original.data_ptr(),
            None if target is None else target.data_ptr(), dt, stage, int(scan_stage), self.plan_ptr)
        return status == 0

    def _check(self, state):
        if (state.shape != self.shape or state.dtype != torch.float64 or state.device.type != 'cpu'
                or not state.movedim(-1, 0).is_contiguous()):
            raise ValueError(f'expected mapped CPU FP64 SoA state {self.shape}')

    def _limit(self, cfl):
        adv, visc = self.scan_values
        if not (np.isfinite(self.scan_values).all() and adv > 0 and visc > 0):
            return False, float('nan')
        return True, float(min(cfl/(7*adv), cfl/(256*visc)))

    def scan(self, state, cfl, *, t=0.):
        self._check(state)
        if self._rhs_buffer is None:
            self._rhs_buffer = torch.empty_like(state)
            self._stage_a = torch.empty_like(state)
            self._stage_b = torch.empty_like(state)
        if not self._evaluate(state, t):
            return False, float('nan')
        self._scanned = state.data_ptr(), t
        return self._limit(cfl)

    def _filter(self, state):
        if self.filter is None:
            return True, False
        weights, matrix, sensor, stats, strength, hits = self.filter
        status = self.lib.mapped_hex_modal_filter(
            self.shape[0], state.data_ptr(), self.jacobian.ctypes.data, weights.ctypes.data,
            matrix.ctypes.data, sensor.ctypes.data, 1, .1, self.threads, stats.ctypes.data,
            strength.ctypes.data, hits.ctypes.data)
        return status == 0, stats[0] > 0

    def advance_finish(self, state, t, dt, cfl, *, out):
        """Return a smaller retry step after a filtered trial fails or exceeds its CFL."""
        self._check(state)
        self._check(out)
        if out.data_ptr() == state.data_ptr():
            raise ValueError('RK output aliases its input')
        if self._scanned != (state.data_ptr(), t) and not self.scan(state, cfl, t=t)[0]:
            return out, False, float('nan')
        self._scanned = None
        a, b = self._stage_a, self._stage_b
        self.lib.mapped_hex_stage(state.numel(), state.data_ptr(), a.data_ptr(),
                                   self._rhs_buffer.data_ptr(), dt, 0, self.threads)
        valid, filtered = self._filter(a)
        if not valid:
            return out, False, .5*dt if filtered else float('nan')
        for stage, (source, target, fraction) in enumerate(
                ((a,b,.5), (b,a,1.), (a,out,.5)), 1):
            valid = self._evaluate(source, t+fraction*dt, original=state, target=target,
                                   dt=dt, stage=stage, scan_stage=filtered)
            if not valid:
                return out, False, .5*dt if filtered else float('nan')
            # The filter can increase the WALE viscous rate in this state and
            # in its neighbors. Scan the remaining stages of this trial step.
            if filtered:
                valid, limit = self._limit(cfl)
                if not valid:
                    return out, False, .5*dt
                if dt > limit*(1.+1e-12):
                    return out, False, min(.5*dt, .9*limit)
            valid, changed = self._filter(target)
            filtered |= changed
            if not valid:
                return out, False, .5*dt if filtered else float('nan')
        valid, next_dt = self.scan(out, cfl, t=t+dt)
        if not valid and filtered:
            next_dt = .5*dt
        return out, valid, next_dt
