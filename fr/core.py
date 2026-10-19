"""Equation-independent Cartesian tensor-product FR in one to three dimensions."""
from dataclasses import dataclass
import math
import torch
from .operators import operators


@dataclass(frozen=True)
class Face:
    """A state trace and its physical flux in the positive coordinate direction."""
    state: torch.Tensor
    flux: torch.Tensor
    coords: tuple | None = None


def central(minus, plus, axis):
    return .5*(minus.flux+plus.flux)


def rusanov_flux(speed):
    """Build a numerical_flux callback from speed(U,axis)."""
    return lambda minus, plus, axis: .5*(minus.flux+plus.flux)-.5*torch.maximum(speed(minus.state,axis),speed(plus.state,axis))[...,None]*(plus.state-minus.state)


class CartesianFR:
    """Layout [*element_axes, *solution_point_axes, variables].

    flux(U,axis) evaluates volume flux. face_flux(U,axis) evaluates physical
    face flux (defaults to flux). numerical_flux(minus,plus,axis) combines
    Face objects. Alternatively flux_pair(Fminus,Fplus) combines only flux
    tensors. Minus always denotes the geometrically lower-coordinate side.

    boundary(coords,inside,axis,side,t) returns an exterior state, where side
    is -1 or +1. Periodic axes bypass this callback. All fluxes use the
    positive coordinate direction, including the lower domain boundary.
    Optional source(coords,U,t) returns a volume source of the same state shape.
    """
    def __init__(self, cells, bounds, p=3, *, flux, numerical_flux=None,
                 flux_pair=None, face_flux=None, boundary=None, periodic=False,
                 speed=None, source=None, device='cpu'):
        self.cells = tuple(cells)
        self.bounds = tuple(tuple(b) for b in bounds)
        self.dim, self.p, self.device = len(self.cells), p, device
        if self.dim not in (1,2,3) or len(self.bounds) != self.dim:
            raise ValueError('provide 1, 2 or 3 cell counts and matching bounds')
        if any(not isinstance(n,int) or n<1 for n in self.cells) or not isinstance(p,int) or p<0:
            raise ValueError('positive integer cell counts and integer p>=0 required')
        if any(len(b)!=2 or not all(math.isfinite(x) for x in b) or b[1]<=b[0] for b in self.bounds):
            raise ValueError('bounds must be finite increasing pairs')
        self.periodic = (periodic,)*self.dim if isinstance(periodic,bool) else tuple(periodic)
        if len(self.periodic)!=self.dim:
            raise ValueError('one periodic flag per direction required')
        if not all(self.periodic) and boundary is None:
            raise ValueError('nonperiodic axes require an exterior-state boundary callback')
        if (numerical_flux is None) == (flux_pair is None):
            raise ValueError('provide exactly one of numerical_flux or flux_pair')
        self.flux = flux
        self.face_flux = flux if face_flux is None else face_flux
        self.numerical_flux = numerical_flux if numerical_flux is not None else lambda m,p,a: flux_pair(m.flux,p.flux)
        self.boundary, self.speed, self.source = boundary, speed, source
        self.z,self.w,self.D,self.L,self.R,self.gl,self.gr = operators(p,device)
        self.h = tuple((b-a)/n for n,(a,b) in zip(self.cells,self.bounds))
        self.shape = self.cells+(p+1,)*self.dim
        coords=[]
        for axis,(n,(a,b),h) in enumerate(zip(self.cells,self.bounds,self.h)):
            eshape=[1]*(2*self.dim)
            nshape=eshape.copy()
            eshape[axis]=n
            nshape[self.dim+axis]=p+1
            center=a+(torch.arange(n,dtype=torch.float64,device=device)+.5)*h
            coords.append((center.reshape(eshape)+h/2*self.z.reshape(nshape)).expand(self.shape))
        self.coords=tuple(coords)
        weights=torch.ones((p+1,)*self.dim,dtype=torch.float64,device=device)
        for axis in range(self.dim):
            shape=[1]*self.dim
            shape[axis]=p+1
            weights=weights*self.w.reshape(shape)
        self.weights=weights
        # Cache physical face coordinates in each axis's oriented layout.
        self.face_coords=[]
        self.trace_coords=[]
        for axis in range(self.dim):
            perm=self._permutation(axis)[:-1]
            pair=[]
            full_pair=[]
            for side,vector,index in ((-1,self.L,0),(1,self.R,-1)):
                values=[]
                for j,x in enumerate(self.coords):
                    value=self._trace(x.permute(perm),vector)
                    if j==axis:
                        positions=self.bounds[axis][0]+self.h[axis]*(torch.arange(self.cells[axis],dtype=value.dtype,device=device)+(0 if side==-1 else 1))
                        value=positions.reshape((self.cells[axis],)+(1,)*(value.ndim-1)).expand_as(value)
                    values.append(value)
                full_pair.append(tuple(values))
                pair.append(tuple(v[index] for v in values))
            self.face_coords.append(pair)
            self.trace_coords.append(full_pair)

    def _permutation(self, axis):
        rest=[a for a in range(self.dim) if a!=axis]
        return [axis]+rest+[self.dim+axis]+[self.dim+a for a in rest]+[2*self.dim]

    def _trace(self,A,vector):
        return (A.movedim(self.dim,-1)*vector).sum(-1)

    def _face(self,U,axis,coords=None):
        return Face(U,self.face_flux(U,axis),coords)

    def rhs(self,U,t=0.):
        if U.ndim != 2*self.dim+1 or tuple(U.shape[:-1]) != self.shape:
            raise ValueError(f'expected state shape {self.shape} + (variables,)')
        out=torch.zeros_like(U)
        for axis,h in enumerate(self.h):
            perm=self._permutation(axis)
            A=U.permute(perm)
            F=self.flux(A,axis)
            ul,ur=self._trace(A,self.L),self._trace(A,self.R)
            fl,fr=self._trace(F,self.L),self._trace(F,self.R)
            cl,cr=self.trace_coords[axis]
            interior=self.numerical_flux(self._face(ur[:-1],axis,tuple(x[:-1] for x in cr)),self._face(ul[1:],axis,tuple(x[1:] for x in cl)),axis)
            if self.periodic[axis]:
                seam=self.numerical_flux(self._face(ur[-1],axis,tuple(x[-1] for x in cr)),self._face(ul[0],axis,tuple(x[0] for x in cl)),axis)
                left,right=seam,seam
            else:
                coords_l,coords_r=self.face_coords[axis]
                exterior_l=self.boundary(coords_l,ul[0],axis,-1,t)
                exterior_r=self.boundary(coords_r,ur[-1],axis,1,t)
                left=self.numerical_flux(self._face(exterior_l,axis,coords_l),self._face(ul[0],axis,coords_l),axis)
                right=self.numerical_flux(self._face(ur[-1],axis,coords_r),self._face(exterior_r,axis,coords_r),axis)
            common=torch.cat((left[None],interior,right[None]),0)
            deriv=(F.movedim(self.dim,-1)@self.D.T).movedim(-1,self.dim)
            shape=[1]*A.ndim
            shape[self.dim]=self.p+1
            deriv=deriv+self.gl.reshape(shape)*(common[:-1]-fl).unsqueeze(self.dim)
            deriv=deriv+self.gr.reshape(shape)*(common[1:]-fr).unsqueeze(self.dim)
            inverse=[perm.index(i) for i in range(U.ndim)]
            out=out-2/h*deriv.permute(inverse)
        return out if self.source is None else out+self.source(self.coords,U,t)

    def integral(self,U):
        dims=tuple(range(2*self.dim))
        return (U*self.weights[...,None]).sum(dims)*math.prod(self.h)/2**self.dim

    def norm(self,U):
        return (self.integral(U*U)/math.prod(b-a for a,b in self.bounds)).sqrt()

    def timestep(self,U,cfl=.15):
        if self.speed is None:
            raise ValueError('CFL time step requires speed(U,axis) callback; otherwise use fixed dt')
        if not math.isfinite(cfl) or cfl<=0:
            raise ValueError('CFL must be finite and positive')
        rate=sum(self.speed(U,a)/h for a,h in enumerate(self.h)).max().item()
        if not math.isfinite(rate) or rate<0:
            raise ValueError('wave speed must be finite and nonnegative')
        return cfl/((2*self.p+1)*rate) if rate else 1e30
