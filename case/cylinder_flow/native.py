"""CPU custom operators registered through TORCH_LIBRARY and torch.ops."""
import hashlib
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from cylinder import Cylinder

_LOADED = False


def load_operators():
    global _LOADED
    if _LOADED:
        return
    source = Path(__file__).with_name('native.cpp')
    flags = ['-O3','-std=c++20','-march=native','-fopenmp','-ffp-contract=fast','-fno-math-errno']
    key = hashlib.sha256(source.read_bytes()+torch.__version__.encode()+
                         ' '.join(flags).encode()).hexdigest()[:16]
    build = Path('/tmp/fr-cylinder-native')/key
    build.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MAX_JOBS','1')
    load(name='cylinder_flow_native',sources=[str(source)],extra_cflags=flags,extra_ldflags=['-fopenmp'],
         build_directory=str(build),is_python_module=False,verbose=False)
    _LOADED = True


class NativeCylinder(Cylinder):
    native_backend = True
    implementation = 'torch.ops.cylinder_flow CPU FP64 C++'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        load_operators()
        s=self.s; b=s.blocks[0]
        # All static operators retain their existing curved-mesh FP64 values.
        self.native_data = [t.contiguous() for t in (
            b['B'],b['face_B'],self.Gx,self.Gy,self.lift_matrix,s.normals,
            (self.mu*self.penalty*(s.p+1)**2/self.face_h[:,None]).expand_as(s.face_jacobian),self.far,
            self.base_step,b['ids'],s.neighbor_points,self.mesh.face_tags,
            self.h,b['weight'])]
        # Only boundary faces may carry a physical boundary condition.
        tags=self.native_data[11].clone()
        tags[~self.mesh.boundary]=0
        self.native_data[11]=tags
        self.diffusion=self.viscous_step_factor/(16*self.penalty*(s.p+1)**4*
                       self.mu*max(4/3,1.4/self.prandtl))

    def rhs(self,U):
        return torch.ops.cylinder_flow.rhs(U,self.native_data,self.mu,self.prandtl,self.diffusion)

    def valid(self,U):
        return torch.ops.cylinder_flow.valid(U,self.native_data,self.mu,self.prandtl,self.diffusion)

    def residual_norm(self,F):
        return torch.ops.cylinder_flow.residual_norm(F,self.native_data,self.mu,self.prandtl,self.diffusion)

    def make_step_with_rhs(self,time_mode='global',*,compile=False):
        if time_mode not in ('global','local'):
            raise ValueError('unknown time mode')
        def advance(U,cfl,first_rhs):
            return torch.ops.cylinder_flow.step(U,cfl,first_rhs,self.native_data,
                self.mu,self.prandtl,self.diffusion,time_mode=='local')
        return advance
