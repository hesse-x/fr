"""Independent Torch reference flux and SSPRK(4,3) for correctness tests."""
import torch
from fr import Euler


def ssprk43_add(rhs, U, t, dt):
    """SSPRK(4,3) using Torch's scaled add to avoid full-field products."""
    u1 = torch.add(U, rhs(U, t), alpha=dt/2)
    u2 = torch.add(u1, rhs(u1, t+dt/2), alpha=dt/2)
    u3 = torch.add(U*(2/3), u2, alpha=1/3)
    u3.add_(rhs(u2, t+dt), alpha=dt/6)
    return torch.add(u3, rhs(u3, t+dt/2), alpha=dt/2)


class ExplicitEuler(Euler):
    """Component-explicit flux keeps the component index static in the graph.

    The broadcast form U*vel plus masked slice additions lets Inductor fuse
    the five component expressions into a runtime component loop whose body
    selects the pressure term with integer compares and blends. Building
    each component explicitly removes that predication; the arithmetic per
    element is unchanged.
    """
    def flux(self, U, axis):
        rho, velocity, pressure = self.primitive(U)
        vel = velocity[..., axis]
        momenta = [U[..., i+1]*vel for i in range(self.dim)]
        momenta[axis] = momenta[axis]+pressure
        return torch.stack((U[..., 0]*vel, *momenta,
                            U[..., -1]*vel+pressure*vel), -1)
