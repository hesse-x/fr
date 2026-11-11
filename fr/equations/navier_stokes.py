"""Calorically perfect gas, constant viscosity and Fourier heat conduction.

R=1, Stokes hypothesis (bulk viscosity zero). Gradients end in
(velocity components plus temperature, spatial direction).
"""
import math
import torch
from .euler import Euler


class NavierStokes(Euler):
    def __init__(self, dim=3, gamma=1.4, mu=1/1600, prandtl=.71):
        super().__init__(dim, gamma)
        if not all(math.isfinite(v) for v in (gamma, mu, prandtl)) or mu <= 0 or prandtl <= 0:
            raise ValueError('finite gamma>1, mu>0 and prandtl>0 required')
        self.mu, self.prandtl = mu, prandtl
        self.kappa = mu*gamma/(gamma-1)/prandtl

    def flux(self, U, axis):
        # Explicit components match the native five-component arithmetic.
        _, velocity, pressure = self.primitive(U)
        vel = velocity[..., axis]
        momenta = [U[..., i+1]*vel for i in range(self.dim)]
        momenta[axis] = momenta[axis]+pressure
        return torch.stack((U[..., 0]*vel, *momenta,
                            U[..., -1]*vel+pressure*vel), -1)

    def primitive_variables(self, U):
        rho, velocity, pressure = self.primitive(U)
        return torch.cat((velocity, (pressure/rho)[..., None]), -1)

    def viscous_flux(self, U, gradient):
        if gradient.shape != U.shape[:-1]+(self.dim+1, self.dim):
            raise ValueError('expected gradient of velocity and temperature')
        primitive = self.primitive_variables(U)
        du = gradient[..., :self.dim, :]
        stress = self.mu*(du+du.transpose(-1, -2)
                         -2/3*du.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
                         *torch.eye(self.dim, dtype=U.dtype, device=U.device))
        energy = (primitive[..., :self.dim, None]*stress).sum(-2)+self.kappa*gradient[..., self.dim, :]
        return torch.cat((torch.zeros_like(energy[..., None, :]), stress,
                          energy[..., None, :]), -2)

    def conservative(self, rho, velocity, pressure):
        if velocity.shape[-1] != self.dim:
            raise ValueError('velocity dimension does not match equation')
        rho, pressure = torch.broadcast_tensors(torch.as_tensor(rho, dtype=velocity.dtype, device=velocity.device),
                                                torch.as_tensor(pressure, dtype=velocity.dtype, device=velocity.device))
        rho = torch.broadcast_to(rho, velocity.shape[:-1])
        pressure = torch.broadcast_to(pressure, velocity.shape[:-1])
        energy = pressure/(self.gamma-1)+.5*rho*velocity.square().sum(-1)
        return torch.cat((rho[..., None], rho[..., None]*velocity, energy[..., None]), -1)

    def timestep(self, grid, U, cfl=.1):
        if not math.isfinite(cfl) or cfl <= 0:
            raise ValueError('positive finite CFL required')
        density, velocity, pressure = self.primitive(U)
        rho = float(density.min())
        diffusivity = max(4/3*self.mu, self.kappa*(self.gamma-1))/rho
        viscous = cfl/((grid.p+1)**4*sum(float(h.min())**-2 for h in grid.widths)*diffusivity)
        sound = (self.gamma*pressure/density).sqrt()
        rate = sum((velocity[..., a].abs()+sound)/grid.cell_width(a, variables=False)
                   for a in range(self.dim)).max()
        return min(cfl/((2*grid.p+1)*float(rate)), viscous)
