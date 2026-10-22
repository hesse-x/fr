"""Experimental compressible laminar NS on curved triangles (BR1 + penalty)."""
import importlib.util
from pathlib import Path
import math
import torch

spec = importlib.util.spec_from_file_location('naca_euler_physics', Path(__file__).parents[1]/'naca0012-euler/airfoil.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Airfoil(module.Airfoil):
    def __init__(self, mesh, p=2, mach=.1, alpha=0., quadrature_order=6,
                 reynolds=1000., prandtl=.72, penalty=4., viscous_step_factor=1.):
        if not all(math.isfinite(v) and v > 0 for v in (reynolds, prandtl, penalty, mach, viscous_step_factor)):
            raise ValueError('positive finite Re, Pr, penalty and Mach required')
        super().__init__(mesh, p, mach, alpha, quadrature_order)
        self.reynolds, self.prandtl, self.penalty = reynolds, prandtl, penalty
        # rho_inf=c_inf=chord=1; Re=rho_inf*u_inf*chord/mu.
        self.mu = mach/reynolds
        self.viscous_step_factor = viscous_step_factor
        b = self.s.blocks[0]
        perimeter = (self.s.face_jacobian[b['ids']]*self.s.fw).sum((1,2))
        self.h = 2*b['areas']/perimeter
        hf = torch.empty(self.mesh.nfaces, dtype=torch.float64)
        hf[b['ids']] = self.h[:,None].expand_as(b['ids'])
        self.face_h = torch.minimum(hf, hf[self.s.neighbor])

    def primitive_vector(self, U):
        rho, vel, pressure = self.eq.primitive(U)
        # theta=p/rho=R*T, so conductivity for grad(theta) is mu*gamma/((gamma-1)*Pr).
        return torch.cat((vel, (pressure/rho)[...,None]), -1)

    def surface(self, F):
        s = self.s; b = s.blocks[0]
        return torch.einsum('fqi,efqv,q,efq->eiv', b['face_B'], F[b['ids']], s.fw, s.face_jacobian[b['ids']])

    def _interpolate_primitives(self, W):
        """Compute volume and both face traces once per viscous residual."""
        s = self.s; b = s.blocks[0]
        V = W.reshape(b['ne'], b['ns'], -1)
        Q = b['B']@V
        L = s._states(W); R = s.neighbor_states(L)
        return Q, L, R

    def gradient(self, W):
        return self._gradient_from_traces(*self._interpolate_primitives(W))

    def _gradient_from_traces(self, Q, L, R):
        s = self.s; b = s.blocks[0]
        trace = .5*(L+R)
        trace[self.wall_ids,:,:2] = 0.  # no slip; theta trace is interior (adiabatic)
        trace[self.outer_ids] = self.primitive_vector(self.far).expand_as(trace[self.outer_ids])
        grad = []
        for axis, G in enumerate((self.Gx, self.Gy)):
            F = trace*s.normals[...,axis,None]
            grad.append(-G@Q-self.lift_matrix@F[b['ids']].reshape(b['ne'],-1,3))
        return torch.stack(grad,-1).reshape(s.ndof,3,2)

    def viscous_flux(self, W, grad):
        g = grad[...,:2,:]
        div = g[...,0,0]+g[...,1,1]
        stress = self.mu*(g+g.transpose(-1,-2)-2/3*div[...,None,None]*torch.eye(2,dtype=W.dtype))
        heat = self.mu*3.5/self.prandtl*grad[...,2,:]
        energy = torch.einsum('...i,...ij->...j',W[...,:2],stress)+heat
        return torch.cat((torch.zeros_like(energy[...,None,:]),stress,energy[...,None,:]),-2)

    def viscous(self, U):
        s = self.s; b = s.blocks[0]
        W = self.primitive_vector(U)
        Q, L, R = self._interpolate_primitives(W)
        grad = self._gradient_from_traces(Q, L, R)
        G = grad.reshape(b['ne'],b['ns'],6)
        volume_flux = self.viscous_flux(Q,(b['B']@G).reshape(b['ne'],-1,3,2))
        gl = s._states(grad.reshape(s.ndof,6)).reshape(*L.shape,2)
        gr = s.neighbor_states(gl.flatten(-2)).reshape_as(gl)
        flux = .5*(self.viscous_flux(L,gl)+self.viscous_flux(R,gr))
        common = torch.einsum('fqva,fqa->fqv',flux,s.normals)
        jump = R-L
        jump[self.outer_ids] = self.primitive_vector(self.far)-L[self.outer_ids]
        jump[self.wall_ids,:,:2] = -L[self.wall_ids,:,:2]
        jump[self.wall_ids,:,2] = 0.
        scale = self.penalty*(s.p+1)**2/self.face_h[:,None]
        common[...,1:3] += self.mu*scale[...,None]*jump[...,:2]
        common[...,3] += self.mu*3.5/self.prandtl*scale*jump[...,2]
        # Stationary adiabatic wall: zero total viscous energy flux.
        common[self.wall_ids,:,3] = 0.
        value = -self.Gx@volume_flux[...,0]-self.Gy@volume_flux[...,1]-self.lift_matrix@common[b['ids']].reshape(b['ne'],-1,4)
        return value.reshape_as(U), common

    def rhs(self, U):
        return super().rhs(U)+self.viscous(U)[0]

    def make_step_with_rhs(self, time_mode='global', *, compile=False):
        """Reuse RHS(U) as stage one and return RHS of the accepted candidate.

        After the initial RHS evaluation this uses four new RHS evaluations per
        complete SSPRK(4,3) step, including the next state's residual for logging
        and reuse by the following step. A rejected candidate discards its RHS.
        """
        if time_mode not in ('global', 'local'):
            raise ValueError('unknown time mode')

        def step(U, cfl, first_rhs):
            dt = self.local_dt(U, cfl)
            if time_mode == 'global':
                dt = dt.amin().expand_as(dt)
            V = U.reshape(-1, self.s.blocks[0]['ns'], 4)

            def f(X):
                return self.rhs(X.reshape_as(U)).reshape_as(V)

            u1 = V+.5*dt*first_rhs.reshape_as(V)
            u2 = u1+.5*dt*f(u1)
            u3 = 2/3*V+1/3*u2+dt/6*f(u2)
            candidate = (u3+.5*dt*f(u3)).reshape_as(U)
            return candidate, dt, self.rhs(candidate)

        return torch.compile(step, fullgraph=True) if compile else step

    def local_dt(self, U, cfl):
        convective = super().local_dt(U,cfl)
        b = self.s.blocks[0]
        rho = (b['B']@U.reshape(b['ne'],b['ns'],4))[...,0].amin(1)
        diffusive = self.viscous_step_factor*cfl*self.h**2*rho/(16*self.penalty*(self.s.p+1)**4*self.mu*max(4/3,1.4/self.prandtl))
        return torch.minimum(convective,diffusive[:,None,None])

    def loads(self, U):
        cl, cd, pressure = super().loads(U)
        _, flux = self.viscous(U)
        ids = self.wall_ids; s = self.s
        force = -(flux[ids,:,1:3]*s.face_jacobian[ids,:,None]*s.fw[None,:,None]).sum((0,1))/(.5*self.mach**2)
        return cl+float(force@torch.stack((-self.direction[1],self.direction[0]))), cd+float(force@self.direction), pressure

    def wall_data(self, U):
        """Numerical traction consistent with momentum balance; tangent points toward +x."""
        s = self.s; ids = self.wall_ids
        clp, cdp, pressure = super().loads(U)
        _, flux = self.viscous(U)
        traction = -flux[ids,:,1:3]
        n = s.normals[ids]
        orient = torch.where(n[...,1]>=0,1.,-1.)
        tangent = torch.stack((n[...,1],-n[...,0]),-1)*orient[...,None]
        q = .5*self.mach**2
        cf = (traction*tangent).sum(-1)/q
        force = (traction*s.face_jacobian[ids,:,None]*s.fw[None,:,None]).sum((0,1))/q
        cdv = float(force@self.direction)
        clv = float(force@torch.stack((-self.direction[1],self.direction[0])))
        return dict(CL=clp+clv,CD_total=cdp+cdv,CL_pressure=clp,CL_viscous=clv,
                    CD_pressure=cdp,CD_viscous=cdv), torch.cat((s.face_coordinates[ids],
                    ((pressure-1/1.4)/q)[...,None],cf[...,None],
                    s.normals[ids],traction/q),-1).reshape(-1,8)
