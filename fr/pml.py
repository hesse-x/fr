"""Directional split PML for constant-coefficient, zero-mean-flow systems.

For uniform flow use a translating coordinate frame; the layer then moves too.
This is not a stationary-boundary convected-flow PML.
"""
import math
import torch
from .core import Face


class SplitPML:
    def __init__(self, equation, inner_bounds, thickness, reflection=1e-6, order=3):
        if thickness <= 0 or not 0 < reflection < 1 or order < 1:
            raise ValueError('positive thickness/order and reflection in (0,1) required')
        self.equation, self.dim = equation, len(inner_bounds)
        self.inner_bounds, self.thickness = inner_bounds, thickness
        self.sigma_max = -(order+1)*equation.c*math.log(reflection)/(2*thickness)
        self.order = order

    def physical(self, Q):
        return Q.reshape(*Q.shape[:-1], self.dim, -1).sum(-2)

    def pack(self, U):
        return (U.unsqueeze(-2).expand(*U.shape[:-1], self.dim, U.shape[-1])/self.dim).reshape(*U.shape[:-1], -1).clone()

    def lift(self, F, axis):
        result = torch.zeros(*F.shape[:-1], self.dim, F.shape[-1], dtype=F.dtype, device=F.device)
        result[..., axis, :] = F
        return result.flatten(-2)

    def flux(self, Q, axis):
        return self.lift(self.equation.flux(self.physical(Q), axis), axis)

    def numerical_flux(self, physical_flux):
        def callback(m, p, axis):
            um, up = self.physical(m.state), self.physical(p.state)
            return self.lift(physical_flux(Face(um, self.equation.flux(um, axis),m.coords), Face(up, self.equation.flux(up, axis),p.coords), axis), axis)
        return callback

    def rates(self, coords):
        return torch.stack([self.sigma_max*(torch.maximum(a-x, x-b).clamp(min=0)/self.thickness).clamp(max=1)**self.order for x, (a,b) in zip(coords,self.inner_bounds)], -1)

    def source(self, coords, Q, t):
        split = Q.reshape(*Q.shape[:-1], self.dim, -1)
        return (-self.rates(coords)[..., :, None]*split).flatten(-2)

    def boundary(self, physical_boundary):
        def callback(coords, Q, axis, side, t):
            U = self.physical(Q)
            ghost = physical_boundary(coords, U, axis, side, t)
            return Q+self.pack(ghost-U)
        return callback
