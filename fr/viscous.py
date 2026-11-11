"""Configurable NS viscous operators, independent of case initial/load data.

Both implementations expose gradient(W,t), flux(W,G), evaluate(U,t) and
rhs(U,t). evaluate returns (volume residual, outward numerical face flux);
the second entry is None on Cartesian grids. Boundary callbacks prescribe
primitive values, not Euler ghost states. Mesh callbacks can additionally
replace the normal flux, e.g. zero energy flux on an adiabatic wall.
"""
from dataclasses import dataclass
import math
import torch

from .core import CartesianFR
from .second_order import CartesianLDG, CartesianBR1Penalty
from .unstructured import MeshFR2D
from .equations.navier_stokes import NavierStokes


@dataclass(frozen=True)
class ViscousConfig:
    scheme: str = 'ldg'
    penalty: float = 4.

    def __post_init__(self):
        if self.scheme not in ('ldg', 'br1-penalty'):
            raise ValueError('viscous scheme must be ldg or br1-penalty')
        if not math.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError('positive finite penalty required')

    @property
    def timestep_factor(self):
        # Conservative penalty-dependent estimate, not a sharp CFL bound.
        return max(1., self.penalty) if self.scheme == 'br1-penalty' else 1.


class CartesianViscous:
    def __init__(self, grid, equation, config, boundary=None, normal_flux=None):
        self.grid, self.equation, self.config = grid, equation, config
        if normal_flux is not None:
            raise ValueError('Cartesian normal-flux callback is not supported')
        if config.scheme == 'ldg':
            self.derivatives = CartesianLDG(grid, boundary)
        else:
            self.derivatives = CartesianBR1Penalty(grid, boundary, penalty=config.penalty)

    def gradient(self, W, t=0.):
        return self.derivatives.gradient(W, t)

    def flux(self, W, gradient):
        return self.equation.viscous_flux_from_primitive(W, gradient)

    def evaluate(self, U, t=0.):
        eq = self.equation
        W = eq.primitive_variables(U)
        flux = self.flux(W, self.gradient(W, t))
        return self.divergence(U, flux, t), None

    def divergence(self, U, flux, t=0.):
        eq = self.equation
        result = self.derivatives.divergence(flux, t)
        if self.config.scheme == 'br1-penalty':
            W = eq.primitive_variables(U)
            jump = self.derivatives.penalty_divergence(W, t)
            result[..., 1:1+eq.dim] += eq.mu*jump[..., :eq.dim]
            result[..., -1] += eq.kappa*jump[..., -1]
        return result

    def rhs(self, U, t=0.):
        return self.evaluate(U, t)[0]


class MeshViscous:
    """Curved triangular/quad/mixed NS, with a paired face orientation for LDG.

    An interior pair selects the smaller face ID for the primal trace and
    the larger ID for the dual flux. The choice is complementary across the
    pair and independent of face quadrature reversal. Exterior traces/fluxes
    come from the boundary callbacks for either scheme.
    """
    def __init__(self, grid, equation, config, boundary=None, normal_flux=None):
        self.grid, self.equation, self.config = grid, equation, config
        self.boundary, self.normal_flux = boundary, normal_flux
        if grid.boundary_groups and boundary is None:
            raise ValueError('mesh viscous boundaries require a primitive value callback')
        self.blocks = []
        face_h = torch.empty_like(grid.face_jacobian[:, 0])
        for b in grid.blocks:
            inv = torch.linalg.inv(b['mass'])
            lift = -torch.einsum('eij,fqj,q,efq->efiq', inv, b['face_B'],
                                 grid.fw, grid.face_jacobian[b['ids']])
            perimeter = (grid.face_jacobian[b['ids']]*grid.fw).sum((1, 2))
            h = 2*b['areas']/perimeter
            face_h[b['ids']] = h[:, None].expand_as(b['ids'])
            self.blocks.append(dict(Gx=inv@b['Gx'], Gy=inv@b['Gy'],
                                    lift=lift.permute(0, 2, 1, 3).flatten(2).contiguous(), h=h))
        self.face_h = torch.minimum(face_h, face_h[grid.neighbor])
        ids = torch.arange(grid.mesh.nfaces, device=grid.neighbor.device)
        self.primal_neighbor = ids > grid.neighbor

    def _boundary_values(self, trace, inside, t):
        s = self.grid
        for tag, ids in s.boundary_groups:
            xy = s.face_coordinates[ids]
            trace[ids] = self.boundary(tuple(xy.unbind(-1)), inside[ids], s.normals[ids], tag, t)
        return trace

    def _gradient_from_traces(self, W, L, R, t):
        s = self.grid
        trace = self._primal_trace(L, R)
        trace = self._boundary_values(trace, L, t)
        gradient = W.new_empty(*W.shape, 2)
        nv = W.shape[-1]
        for b, geometry in zip(s.blocks, self.blocks):
            Q = b['B']@W[b['offset']].reshape(b['ne'], b['ns'], nv)
            values = []
            for axis, name in enumerate(('Gx', 'Gy')):
                F = trace*s.normals[..., axis, None]
                values.append(-geometry[name]@Q-geometry['lift']@F[b['ids']].reshape(b['ne'], -1, nv))
            gradient[b['offset']] = torch.stack(values, -1).reshape(-1, nv, 2)
        return gradient

    def gradient(self, W, t=0.):
        L = self.grid._states(W)
        return self._gradient_from_traces(W, L, self.grid.neighbor_states(L), t)

    def flux(self, W, gradient):
        return self.equation.viscous_flux_from_primitive(W, gradient)

    def evaluate(self, U, t=0.):
        s, eq = self.grid, self.equation
        W = eq.primitive_variables(U)
        L = s._states(W)
        R = s.neighbor_states(L)
        grad = self._gradient_from_traces(W, L, R, t)
        gl = s._states(grad.flatten(-2)).reshape(*L.shape, 2)
        gr = s.neighbor_states(gl.flatten(-2)).reshape_as(gl)
        fl, fr = self.flux(L, gl), self.flux(R, gr)
        flux = self._dual_flux(fl, fr)
        common = torch.einsum('fqva,fqa->fqv', flux, s.normals)
        common = self._penalize(common, L, R, t)
        if self.normal_flux is not None:
            for tag, ids in s.boundary_groups:
                xy = s.face_coordinates[ids]
                common[ids] = self.normal_flux(tuple(xy.unbind(-1)), L[ids],
                    s.normals[ids], tag, t, common[ids])
        result = torch.empty_like(U)
        for b, geometry in zip(s.blocks, self.blocks):
            Q = b['B']@W[b['offset']].reshape(b['ne'], b['ns'], 3)
            G = (b['B']@grad[b['offset']].reshape(b['ne'], b['ns'], 6)).reshape(b['ne'], -1, 3, 2)
            volume = self.flux(Q, G)
            value = (-geometry['Gx']@volume[..., 0]-geometry['Gy']@volume[..., 1]
                     -geometry['lift']@common[b['ids']].reshape(b['ne'], -1, 4))
            result[b['offset']] = value.reshape(-1, 4)
        return result, common

    def rhs(self, U, t=0.):
        return self.evaluate(U, t)[0]


class MeshLDGViscous(MeshViscous):
    def _primal_trace(self, L, R):
        return torch.where(self.primal_neighbor[:, None, None], R, L)

    def _dual_flux(self, fl, fr):
        flux = torch.where(self.primal_neighbor[:, None, None, None], fl, fr)
        # Exterior dual flux is interior; boundary callbacks may replace it.
        ids = self.grid.mesh.boundary.to(fl.device)
        flux[ids] = fl[ids]
        return flux

    def _penalize(self, common, L, R, t):
        return common


class MeshBR1PenaltyViscous(MeshViscous):
    def _primal_trace(self, L, R):
        return .5*(L+R)

    def _dual_flux(self, fl, fr):
        return .5*(fl+fr)

    def _penalize(self, common, L, R, t):
        s, eq = self.grid, self.equation
        exterior = self._boundary_values(R.clone(), L, t)
        jump = exterior-L
        scale = self.config.penalty*(s.p+1)**2/self.face_h[:, None]
        common[..., 1:3] += eq.mu*scale[..., None]*jump[..., :2]
        common[..., 3] += eq.kappa*scale*jump[..., 2]
        return common


def make_viscous_operator(grid, equation, *, scheme='ldg', penalty=4.,
                          boundary=None, normal_flux=None):
    """Create an NS operator; geometry/backend are independent of scheme choice."""
    config = ViscousConfig(scheme, penalty)
    if not isinstance(equation, NavierStokes):
        raise TypeError('NS viscous operator requires a NavierStokes equation')
    if equation.dim != grid.dim:
        raise ValueError('equation and grid dimensions must match')
    if isinstance(grid, CartesianFR):
        return CartesianViscous(grid, equation, config, boundary, normal_flux)
    if isinstance(grid, MeshFR2D):
        implementation = MeshLDGViscous if scheme == 'ldg' else MeshBR1PenaltyViscous
        return implementation(grid, equation, config, boundary, normal_flux)
    raise TypeError('expected CartesianFR or MeshFR2D')
