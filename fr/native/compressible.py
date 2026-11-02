"""Shared CPU FP64 native Euler/BR1 NS kernels for homogeneous curved meshes."""
import hashlib
import os
import weakref
from pathlib import Path

import torch
from torch.utils.cpp_extension import load
from . import header_bytes

_LOADED = False


def load_operators():
    global _LOADED
    if _LOADED:
        return
    source = Path(__file__).with_suffix('.cpp')
    flags = ['-O3', '-std=c++20', '-march=native', '-fopenmp',
             '-ffp-contract=fast', '-fno-math-errno']
    key = hashlib.sha256(source.read_bytes()+header_bytes()+torch.__version__.encode()+
                         ' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-compressible-native')/key
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MAX_JOBS', '1')
    # Keep the existing namespace for cylinder callers; all three cases share it.
    load(name='fr_compressible_native', sources=[str(source)], extra_cflags=flags,
         extra_ldflags=['-fopenmp'], build_directory=str(build),
         is_python_module=False, verbose=False)
    _LOADED = True


class NativeCompressible:
    """Native solver over geometry/load data; calls per instance are sequential."""
    native_backend = True
    implementation = 'shared native Euler/BR1 NS CPU FP64 C++'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_operators()
        s = self.s
        b = s.blocks[0]
        perimeter = (s.face_jacobian[b['ids']]*s.fw).sum((1, 2))
        h = 2*b['areas']/perimeter
        if hasattr(self, 'mu'):
            mu = self.mu
            penalty = (mu*self.penalty*(s.p+1)**2/self.face_h[:, None]).expand_as(s.face_jacobian)
            diffusion = self.viscous_step_factor/(16*self.penalty*(s.p+1)**4*
                        mu*max(4/3, 1.4/self.prandtl))
        else:
            mu = 0.
            penalty = torch.zeros_like(s.face_jacobian)
            diffusion = float('inf')
        self._native_parameters = (mu, getattr(self, 'prandtl', .72), diffusion)
        self.native_data = [t.contiguous() for t in (
            b['B'], b['face_B'], self.Gx, self.Gy, self.lift_matrix, s.normals,
            penalty, self.far, self.base_step, b['ids'], s.neighbor_points,
            self.mesh.face_tags, h, b['weight'])]
        tags = self.native_data[11].clone()
        tags[~self.mesh.boundary] = 0
        self.native_data[11] = tags
        ne, ns = b['ne'], b['ns']
        faces = s.normals.shape[0]*s.normals.shape[1]
        self._native_workspace = [torch.empty(count,dtype=torch.float64) for count in
            (ne*ns*3 if mu else 0, faces*4, faces*3 if mu else 0,
             ne*ns*6 if mu else 0, faces*6 if mu else 0)]
        self._native_scan = None

    def _cached_scan(self, U):
        cached = self._native_scan
        if cached is not None and cached[0]() is U and cached[1] == U._version:
            return cached[2]
        return None

    def rhs(self, U):
        return torch.ops.cylinder_flow.rhs_cached(U, self.native_data,
            *self._native_parameters, self._native_workspace)

    def valid(self, U):
        valid, scan = torch.ops.cylinder_flow.scan_state(U, self.native_data, *self._native_parameters)
        self._native_scan = (weakref.ref(U), U._version, scan) if valid and not U.is_inference() else None
        return valid

    def residual_norm(self, F):
        return torch.ops.cylinder_flow.residual_norm(F, self.native_data, *self._native_parameters)

    def make_step(self, method='ssprk43', time_mode='global'):
        if method not in ('ssprk43', 'rk4') or time_mode not in ('global', 'local'):
            raise ValueError('unknown RK method or time mode')
        def advance(U, cfl):
            return torch.ops.cylinder_flow.advance_cached(U, cfl, self.native_data,
                *self._native_parameters, time_mode == 'local', method == 'rk4',
                self._native_workspace, self._cached_scan(U))
        return advance

    def make_step_with_rhs(self, time_mode='global'):
        if time_mode not in ('global', 'local'):
            raise ValueError('unknown time mode')
        def advance(U, cfl, first_rhs):
            return torch.ops.cylinder_flow.step_cached(U, cfl, first_rhs, self.native_data,
                *self._native_parameters, time_mode == 'local', self._native_workspace,
                self._cached_scan(U))
        return advance
