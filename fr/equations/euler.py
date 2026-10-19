"""Optional equation implementations; the FR core has no Euler dependency."""
import torch

GAMMA = 1.4


class Euler:
    def __init__(self, dim, gamma=GAMMA):
        if dim not in (1, 2, 3) or gamma <= 1:
            raise ValueError('Euler requires dim=1/2/3 and gamma>1')
        self.dim, self.gamma, self.nvars = dim, gamma, dim+2

    def primitive(self, U):
        if U.shape[-1] != self.dim+2:
            raise ValueError('Euler state must contain rho, dim momenta, E')
        rho = U[..., 0]
        velocity = U[..., 1:-1]/rho[..., None]
        pressure = (self.gamma-1)*(U[..., -1]-.5*rho*(velocity**2).sum(-1))
        return rho, velocity, pressure

    def flux(self, U, axis):
        _, velocity, pressure = self.primitive(U)
        vel = velocity[..., axis]
        result = U*vel[..., None]
        result = result.clone()
        result[..., axis+1] += pressure
        result[..., -1] += pressure*vel
        return result

    def speed(self, U, axis):
        rho, velocity, pressure = self.primitive(U)
        return velocity[..., axis].abs()+(self.gamma*pressure/rho).sqrt()

    def normal_speed(self,U,normal):
        rho,velocity,pressure=self.primitive(U)
        return (velocity*normal).sum(-1).abs()+(self.gamma*pressure/rho).sqrt()

    def valid(self, U):
        rho, _, pressure = self.primitive(U)
        return bool(torch.isfinite(U).all() and (rho>0).all() and (pressure>0).all())


# Compatibility with the original 2-D benchmark API.
def primitive(U):
    rho, vel, pressure = Euler(2).primitive(U)
    return rho, vel[..., 0], vel[..., 1], pressure


def flux(U, axis):
    equation = Euler(2)
    return equation.flux(U, axis), equation.speed(U, axis)


def rusanov(L, R, axis):
    equation = Euler(2)
    return .5*(equation.flux(L,axis)+equation.flux(R,axis))-.5*torch.maximum(equation.speed(L,axis),equation.speed(R,axis))[...,None]*(R-L)
