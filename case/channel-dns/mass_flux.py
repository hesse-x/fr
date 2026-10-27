"""Constant mass flux via a spatially uniform pressure-gradient Lagrange multiplier."""
import torch


class ConstantMassFlux:
    """Cancel the integrated streamwise momentum RHS at every RK stage.

    The base RHS must have zero imposed forcing. The controlled quantity is
    volume-mean rho*u, not velocity or a pointwise rescaled state. The uniform
    force per volume G is paired with energy source G*u.
    """
    def __init__(self, model, base_rhs):
        if model.forcing != 0:
            raise ValueError('constant mass flux requires an unforced base RHS')
        self.model, self.base_rhs = model, base_rhs
        self.last_force = None

    def __call__(self, state, t=0.):
        rhs = self.base_rhs(state, t)
        force = -self.model.s.integral(rhs[..., 1:2])[0]/self.model.volume
        rhs[..., 1].add_(force)
        rhs[..., 4].add_(force*(state[..., 1]/state[..., 0]))
        self.last_force = force
        return rhs
