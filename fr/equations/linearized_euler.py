"""Uniform-background, inviscid linearized Euler equations in fixed coordinates.

Primitive perturbation state: (rho', u'_1, ..., u'_dim, p').
Unlike isentropic acoustics this includes the independently advected entropy
mode rho' - p'/c0**2. No background gradients or external sources are included.
"""
import math
import torch


class LinearizedEuler:
    def __init__(self, dim=2, *, mean=None, rho0=1., p0=1/1.4, gamma=1.4):
        if dim not in (1,2,3) or not all(math.isfinite(v) and v>0 for v in (rho0,p0)) or not math.isfinite(gamma) or gamma<=1:
            raise ValueError('dim=1/2/3, positive finite rho0/p0 and gamma>1 required')
        self.mean = (0.,)*dim if mean is None else tuple(mean)
        if len(self.mean)!=dim or not all(math.isfinite(v) for v in self.mean):
            raise ValueError('one finite mean velocity per dimension required')
        self.dim, self.nvars = dim, dim+2
        self.rho0, self.p0, self.gamma = rho0, p0, gamma
        self.c = math.sqrt(gamma*p0/rho0)

    def flux(self, U, axis):
        if U.shape[-1]!=self.nvars or axis not in range(self.dim):
            raise ValueError('state must be (rho perturbation, velocity perturbations, pressure perturbation)')
        F = self.mean[axis]*U
        F = F.clone()
        F[...,0] += self.rho0*U[...,axis+1]
        F[...,axis+1] += U[...,-1]/self.rho0
        F[...,-1] += self.rho0*self.c**2*U[...,axis+1]
        return F

    def speed(self, U, axis):
        return torch.full_like(U[...,0],abs(self.mean[axis])+self.c)

    def absolute_jacobian_action(self, jump, minus, plus, axis):
        a, Z = self.mean[axis], self.rho0*self.c
        positive = .5*(jump[...,-1]+Z*jump[...,axis+1])
        negative = .5*(jump[...,-1]-Z*jump[...,axis+1])
        pressure = abs(a+self.c)*positive+abs(a-self.c)*negative
        result = abs(a)*jump.clone()
        result[...,0] = abs(a)*(jump[...,0]-jump[...,-1]/self.c**2)+pressure/self.c**2
        result[...,axis+1] = (abs(a+self.c)*positive-abs(a-self.c)*negative)/Z
        result[...,-1] = pressure
        return result

    def normal_speed(self,U,normal):
        a=sum(v*normal[...,i] for i,v in enumerate(self.mean))
        return a.abs()+self.c

    def absolute_normal_jacobian_action(self,jump,minus,plus,normal):
        a=sum(v*normal[...,i] for i,v in enumerate(self.mean))
        velocity=(jump[...,1:-1]*normal).sum(-1)
        Z=self.rho0*self.c
        positive=.5*(jump[...,-1]+Z*velocity)
        negative=.5*(jump[...,-1]-Z*velocity)
        pressure=(a+self.c).abs()*positive+(a-self.c).abs()*negative
        speed=((a+self.c).abs()*positive-(a-self.c).abs()*negative)/Z
        out=a.abs()[...,None]*jump
        out[...,0]=a.abs()*(jump[...,0]-jump[...,-1]/self.c**2)+pressure/self.c**2
        out[...,1:-1]+=(speed-a.abs()*velocity)[...,None]*normal
        out[...,-1]=pressure
        return out

    def from_acoustic(self, U):
        """Embed (p', velocities) in the zero-entropy subspace."""
        return torch.cat((U[...,:1]/self.c**2,U[...,1:],U[...,:1]),-1)

    def acoustic_state(self, U):
        return torch.cat((U[...,-1:],U[...,1:-1]),-1)

    def energy(self, U):
        """Acoustic energy density; the independent entropy mode is excluded."""
        return .5*(U[...,-1]**2/(self.rho0*self.c**2)+self.rho0*U[...,1:-1].square().sum(-1))

    @staticmethod
    def valid(U):
        # Perturbations can have either sign; background positivity is checked above.
        return bool(torch.isfinite(U).all())
