"""Small verification adapter using generic mesh operators; no production backend."""
import torch
from fr.time import METHODS, ssprk43


class CompressibleReference:
    def __init__(self, physics):
        self.physics = physics

    def __getattr__(self, name):
        return getattr(self.physics, name)

    def rhs(self, U):
        value = self.s.rhs(U)
        return value+self.viscous(U)[0] if hasattr(self.physics, 'mu') else value

    def valid(self, U):
        return (self.eq.valid(U) and self.eq.valid(self.s._states(U)) and
                all(self.eq.valid(b['B']@U[b['offset']].reshape(b['ne'],b['ns'],4))
                    for b in self.s.blocks))

    def local_dt(self, U, cfl):
        b = self.s.blocks[0]
        rho, velocity, pressure = self.eq.primitive(b['B']@U.reshape(b['ne'],b['ns'],4))
        speed = (velocity.norm(dim=-1)+(1.4*pressure/rho).sqrt()).amax(1)
        dt = (cfl*self.base_step/speed)[:,None,None]
        if hasattr(self.physics, 'mu'):
            diffusion = (self.viscous_step_factor*cfl*self.h**2*rho.amin(1)/
                (16*self.penalty*(self.s.p+1)**4*self.mu*max(4/3,1.4/self.prandtl)))
            dt = torch.minimum(dt,diffusion[:,None,None])
        return dt

    def make_step(self, method='ssprk43', time_mode='global'):
        def step(U, cfl):
            dt = self.local_dt(U,cfl)
            if time_mode == 'global': dt = dt.amin().expand_as(dt)
            V = U.reshape(-1,self.s.blocks[0]['ns'],4)
            f = lambda X,t: self.rhs(X.reshape_as(U)).reshape_as(V)
            return METHODS[method](f,V,0.,dt).reshape_as(U),dt
        return step

    def make_step_with_rhs(self, time_mode='global'):
        def step(U, cfl, first_rhs):
            dt = self.local_dt(U,cfl)
            if time_mode == 'global': dt = dt.amin().expand_as(dt)
            V = U.reshape(-1,self.s.blocks[0]['ns'],4)
            f = lambda X,t: self.rhs(X.reshape_as(U)).reshape_as(V)
            result = ssprk43(f,V,0.,dt,first_rhs=first_rhs.reshape_as(V)).reshape_as(U)
            return result,dt,self.rhs(result)
        return step
