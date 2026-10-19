"""Linear acoustics with optional uniform mean flow in fixed coordinates."""
import math
import torch


class Acoustics:
    def __init__(self, dim=2, rho=1., c=1.):
        if dim not in (1, 2, 3) or rho <= 0 or c <= 0:
            raise ValueError('positive rho, c and dim in 1..3 required')
        self.dim, self.rho, self.c, self.nvars = dim, rho, c, dim+1

    def flux(self, U, axis):
        F = torch.zeros_like(U)
        F[..., 0] = self.rho*self.c**2*U[..., axis+1]
        F[..., axis+1] = U[..., 0]/self.rho
        return F

    def speed(self, U, axis):
        return torch.full_like(U[..., 0], self.c)

    def absolute_jacobian_action(self, jump, minus, plus, axis):
        out = torch.zeros_like(jump)
        out[..., 0] = self.c*jump[..., 0]
        out[..., axis+1] = self.c*jump[..., axis+1]
        return out

    def normal_speed(self,U,normal):
        return torch.full_like(U[...,0],self.c)

    def absolute_normal_jacobian_action(self,jump,minus,plus,normal):
        out=torch.zeros_like(jump)
        out[...,0]=self.c*jump[...,0]
        out[...,1:]=self.c*(jump[...,1:]*normal).sum(-1)[...,None]*normal
        return out

    @staticmethod
    def valid(U):
        return bool(torch.isfinite(U).all())

    def energy(self, U):
        return .5*(U[..., 0]**2/(self.rho*self.c**2)+self.rho*(U[..., 1:]**2).sum(-1))


class ConvectedAcoustics(Acoustics):
    """Isentropic linear acoustics in fixed coordinates, uniform mean flow.

    State is (pressure perturbation, velocity perturbations). The mean enters
    all directional fluxes, including the advected transverse velocity mode.
    """
    def __init__(self, mean=(.3,.2), rho=1., c=1.):
        if not all(math.isfinite(v) for v in mean):
            raise ValueError('finite mean velocity required')
        super().__init__(len(mean),rho,c)
        self.mean = tuple(mean)

    def flux(self, U, axis):
        return super().flux(U,axis)+self.mean[axis]*U

    def speed(self, U, axis):
        return torch.full_like(U[...,0],abs(self.mean[axis])+self.c)

    def absolute_jacobian_action(self, jump, minus, plus, axis):
        a, Z = self.mean[axis], self.rho*self.c
        positive = .5*(jump[...,0]+Z*jump[...,axis+1])
        negative = .5*(jump[...,0]-Z*jump[...,axis+1])
        out = abs(a)*jump.clone()
        out[...,0] = abs(a+self.c)*positive+abs(a-self.c)*negative
        out[...,axis+1] = (abs(a+self.c)*positive-abs(a-self.c)*negative)/Z
        return out

    def normal_speed(self,U,normal):
        a=sum(v*normal[...,i] for i,v in enumerate(self.mean))
        return a.abs()+self.c

    def absolute_normal_jacobian_action(self,jump,minus,plus,normal):
        a=sum(v*normal[...,i] for i,v in enumerate(self.mean))
        Z=self.rho*self.c
        velocity=(jump[...,1:]*normal).sum(-1)
        positive=.5*(jump[...,0]+Z*velocity)
        negative=.5*(jump[...,0]-Z*velocity)
        pressure=(a+self.c).abs()*positive+(a-self.c).abs()*negative
        speed=((a+self.c).abs()*positive-(a-self.c).abs()*negative)/Z
        out=a.abs()[...,None]*jump
        out[...,0]=pressure
        out[...,1:]+=(speed-a.abs()*velocity)[...,None]*normal
        return out
