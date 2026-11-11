"""Cartesian FR compressible NS with configurable viscosity and backend presets."""
import math
import torch
from .core import CartesianFR, rusanov_flux
from .viscous import make_viscous_operator
from .equations.navier_stokes import NavierStokes


class CartesianNavierStokes:
    """3-D constant-property NS; periodic box or stationary isothermal y walls.

    ``rhs`` is the independent Torch reference. ``make_stepper`` selects
    ``torch`` or the CPU FP64 SoA p=3 ``native-p3`` execution preset.
    Flux dealiasing is supported by Torch only; native rejects it explicitly.
    """
    def __init__(self, cells, bounds, p=3, *, mu=1/1600, gamma=1.4, prandtl=.71,
                 boundary='periodic', wall_temperature=None, forcing=0.,
                 device='cpu', edges=None, anti_alias='none', quadrature_order=None,
                 reference_state=None, viscous_scheme='ldg', penalty=4.):
        if len(cells) != 3:
            raise ValueError('CartesianNavierStokes requires a three-dimensional grid')
        if boundary not in ('periodic', 'isothermal-channel'):
            raise ValueError('boundary must be periodic or isothermal-channel')
        if boundary == 'isothermal-channel' and (wall_temperature is None or
                not math.isfinite(wall_temperature) or wall_temperature <= 0):
            raise ValueError('isothermal-channel requires positive finite wall_temperature')
        if not math.isfinite(forcing):
            raise ValueError('finite uniform x force per volume required')
        self.eq = NavierStokes(3, gamma=gamma, mu=mu, prandtl=prandtl)
        self.forcing = forcing
        self.boundary_preset = boundary
        self.temperature = wall_temperature if wall_temperature is not None else 1.
        self.volume = math.prod(b-a for a, b in bounds)
        periodic = (True, True, True) if boundary == 'periodic' else (True, False, True)
        self.s = CartesianFR(cells, bounds, p, flux=self.eq.flux,
                             numerical_flux=rusanov_flux(self.eq.speed), speed=self.eq.speed,
                             periodic=periodic, boundary=self.wall, device=device, edges=edges,
                             anti_alias=anti_alias, quadrature_order=quadrature_order)
        self.viscous_operator = make_viscous_operator(self.s, self.eq,
            scheme=viscous_scheme, penalty=penalty, boundary=self.wall_primitive)
        self.viscous_config = self.viscous_operator.config
        self.viscous_scheme = self.viscous_config.scheme
        self.penalty = self.viscous_config.penalty
        # Compatibility for diagnostics that use the derivative pair directly.
        self.ldg = self.viscous_operator.derivatives
        self.reference_ldg = self.ldg
        def convection(U, t=0., fields=None):
            return self.s.rhs(U, t)
        if reference_state is None:
            reference_state = [1., 0., 0., 0., self.temperature/(gamma-1)]
        reference = self.s.z.new_tensor(reference_state)
        if reference.shape != (5,) or not self.eq.valid(reference):
            raise ValueError('reference_state must be a valid five-component conservative state')
        convection.flux_mode = 'local'
        convection.reference_flux = tuple(self.eq.flux(reference, a) for a in range(3))
        self.convection = convection

    @property
    def mu(self):
        return self.eq.mu

    @property
    def kappa(self):
        return self.eq.kappa

    @staticmethod
    def wall(coords, inside, axis, side, t):
        ghost = inside.clone()
        ghost[..., axis+1] *= -1
        return ghost

    def wall_primitive(self, coords, inside, axis, side, t):
        value = torch.zeros_like(inside)
        value[..., 3] = self.temperature
        return value

    def primitive(self, U):
        return self.eq.primitive_variables(U)

    def viscous_flux(self, U, t=0.):
        gradient = self.viscous_operator.gradient(self.primitive(U), t)
        return self.eq.viscous_flux(U, gradient)

    def _wall_energy_correction(self, U, flux):
        s = self.s
        work = ((U[..., 1:4]/U[..., :1])*flux[..., 1:4, 1]).sum(-1, keepdim=True)
        perm = s._permutation(1)
        oriented = work.permute(perm)
        left, right = s._trace(oriented, s.L), s._trace(oriented, s.R)
        correction = torch.zeros_like(oriented)
        shape = [1]*(oriented.ndim-1)
        shape[s.dim-1] = s.p+1
        correction[0] -= s.gl.reshape(shape)*left[0].unsqueeze(s.dim-1)
        correction[-1] -= s.gr.reshape(shape)*right[-1].unsqueeze(s.dim-1)
        return (2/s.cell_width(1))*correction.permute([perm.index(i) for i in range(U.ndim)])

    def rhs(self, U, t=0.):
        flux = self.viscous_flux(U, t)
        result = self.convection(U, t)+self.viscous_operator.divergence(U, flux, t)
        if self.boundary_preset == 'isothermal-channel':
            result[..., -1:] += self._wall_energy_correction(U, flux)
        result[..., 1] += self.forcing
        result[..., -1] += self.forcing*U[..., 1]/U[..., 0]
        return result

    def timestep(self, U, cfl=.1):
        return self.eq.timestep(self.s, U, cfl, viscous_factor=self.viscous_config.timestep_factor)

    def make_stepper(self, preset='torch', *, constant_mass_flux=False,
                     block=(4, 4, 4), boundary_mode='separate'):
        """Return callable advance(U,t,dt); forcing control is opt-in.

        Native also exposes scan/advance_finish and optional 12-field moments.
        Input states must use SoA strides; use ``soa`` to convert once.
        """
        if preset not in ('torch', 'native-p3'):
            raise ValueError('preset must be torch or native-p3')
        if constant_mass_flux and self.forcing != 0:
            raise ValueError('constant mass flux requires an unforced base RHS')
        if boundary_mode not in ('separate', 'fused'):
            raise ValueError('boundary_mode must be separate or fused')
        if preset == 'torch':
            from .time import ssprk43
            from .sources import ConstantMassFlux
            rhs = ConstantMassFlux(self, self.rhs) if constant_mass_flux else self.rhs
            return lambda U, t, dt: ssprk43(rhs, U, t, dt)
        from .native.cartesian_ns import GroupedFused, NativeLocalRKStepper
        plan = GroupedFused(self, block, boundary_mode=boundary_mode)
        return NativeLocalRKStepper(self, plan, constant_mass_flux=constant_mass_flux)

    @staticmethod
    def soa(U):
        return U.movedim(-1, 0).contiguous().movedim(0, -1)
