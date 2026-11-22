"""Local eddy-viscosity closures for three-dimensional compressible LES."""
import math
import torch


def filter_width_scale(grid, mode='dof'):
    """Scale relative to h_eff/(p+1); gauss-center uses h_eff min|xi|/2.

    h_eff is the cube root of cell volume. A central Gauss node makes the
    experimental center-distance definition zero, so such orders are rejected.
    half-dof instead uses h_eff/[2(p+1)] and is valid for any polynomial order.
    """
    if mode == 'dof':
        return 1.
    if mode == 'half-dof':
        return .5
    if mode != 'gauss-center':
        raise ValueError('filter width must be dof, half-dof or gauss-center')
    if grid.p % 2 == 0:
        raise ValueError('gauss-center filter width is zero for even p; use odd p')
    distance = float(grid.z.abs().min())
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('positive finite Gauss center distance required')
    return .5*(grid.p+1)*distance


def filter_width(grid, mode='dof'):
    """Isotropic model width for a 3-D Cartesian grid (scalar or cellwise tensor)."""
    if grid.dim != 3:
        raise ValueError('SGS filter width requires a three-dimensional grid')
    base = (grid.cell_width(0, variables=False)*grid.cell_width(1, variables=False)
            *grid.cell_width(2, variables=False))**(1/3)/(grid.p+1)
    return base*filter_width_scale(grid, mode)


class EddyViscosity:
    """Deviatoric SGS stress and gradient-diffusion heat flux; no SGS k equation."""
    def __init__(self, turbulent_prandtl=.9):
        if not math.isfinite(turbulent_prandtl) or turbulent_prandtl <= 0:
            raise ValueError('positive finite turbulent Prandtl number required')
        self.turbulent_prandtl = turbulent_prandtl

    def flux(self, equation, U, gradient, delta):
        rho, velocity, _ = equation.primitive(U)
        mu_t = self.viscosity(rho, gradient, delta)
        du = gradient[..., :3, :]
        trace = du.diagonal(dim1=-2, dim2=-1).sum(-1)
        stress = mu_t[..., None, None]*(du+du.transpose(-1, -2)
            -2/3*trace[..., None, None]*torch.eye(3, dtype=U.dtype, device=U.device))
        kappa_t = mu_t*equation.gamma/(equation.gamma-1)/self.turbulent_prandtl
        energy = (velocity[..., :, None]*stress).sum(-2)+kappa_t[..., None]*gradient[..., 3, :]
        return torch.cat((torch.zeros_like(energy[..., None, :]), stress, energy[..., None, :]), -2)


class Smagorinsky(EddyViscosity):
    def __init__(self, cs=.1, turbulent_prandtl=.9):
        super().__init__(turbulent_prandtl)
        if not math.isfinite(cs) or cs < 0:
            raise ValueError('finite nonnegative Smagorinsky coefficient required')
        self.cs = cs

    def viscosity(self, density, gradient, delta):
        du = gradient[..., :3, :]
        strain = .5*(du+du.transpose(-1, -2))
        trace = strain.diagonal(dim1=-2, dim2=-1).sum(-1)
        strain = strain-trace[..., None, None]/3*torch.eye(3, dtype=gradient.dtype, device=gradient.device)
        return density*(self.cs*delta)**2*(2*strain.square().sum((-1, -2))).sqrt()


class Vreman(EddyViscosity):
    """Vreman (2004), isotropic filter; C_v multiplies sqrt(B_beta / alpha:alpha).

    gradient[..., velocity, direction] is the transpose of the paper's alpha.
    Sum of squared column cross products evaluates B_beta without cancellation.
    """
    def __init__(self, cv=.07, turbulent_prandtl=.9):
        super().__init__(turbulent_prandtl)
        if not math.isfinite(cv) or cv < 0:
            raise ValueError('finite nonnegative Vreman coefficient required')
        self.cv = cv

    def viscosity(self, density, gradient, delta):
        du = gradient[..., :3, :]
        norm2 = du.square().sum((-1, -2))
        columns = du.unbind(-2)
        b = sum(torch.linalg.cross(columns[i], columns[j], dim=-1).square().sum(-1)
                for i, j in ((0, 1), (0, 2), (1, 2)))
        denominator = torch.where(norm2 > 0, norm2, torch.ones_like(norm2))
        return density*self.cv*delta**2*(b/denominator).sqrt()


class WALE(EddyViscosity):
    """Nicoud–Ducros WALE, using the traceless symmetric square of grad(u)."""
    def __init__(self, cw=.5, turbulent_prandtl=.9):
        super().__init__(turbulent_prandtl)
        if not math.isfinite(cw) or cw < 0:
            raise ValueError('finite nonnegative WALE coefficient required')
        self.cw = cw

    def viscosity(self, density, gradient, delta):
        du = gradient[..., :3, :]
        strain = .5*(du+du.transpose(-1, -2))
        square = du @ du
        trace = square.diagonal(dim1=-2, dim2=-1).sum(-1)
        sd = .5*(square+square.transpose(-1, -2))-trace[..., None, None]/3*torch.eye(
            3, dtype=gradient.dtype, device=gradient.device)
        s2 = strain.square().sum((-1, -2))
        d2 = sd.square().sum((-1, -2))
        denominator = s2.pow(2.5)+d2.pow(1.25)
        safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        return density*(self.cw*delta)**2*d2.pow(1.5)/safe


_cartesian_filter_width = filter_width


class CartesianLESMixin:
    """SGS configuration, viscous flux and explicit stability bound for Cartesian NS."""
    def __init__(self, *args, sgs_model='vreman', cv=.07, cs=.1, cw=.5, filter_ratio=1.,
                 turbulent_prandtl=.9, filter_width='dof', **kwargs):
        if sgs_model not in ('none', 'vreman', 'smagorinsky', 'wale'):
            raise ValueError('SGS model must be none, vreman, smagorinsky or wale')
        self.sgs_model = sgs_model
        self.sgs = {'wale': lambda: WALE(cw, turbulent_prandtl),
                    'vreman': lambda: Vreman(cv, turbulent_prandtl),
                    'smagorinsky': lambda: Smagorinsky(cs, turbulent_prandtl),
                    'none': lambda: None}[sgs_model]()
        if not math.isfinite(filter_ratio) or filter_ratio <= 0:
            raise ValueError('positive finite filter_ratio required')
        self.filter_ratio = filter_ratio
        if kwargs.get('viscous_scheme', 'ldg') != 'ldg':
            raise ValueError('LES currently requires ldg')
        super().__init__(*args, **kwargs)
        self.filter_width = filter_width
        self.sgs_filter_scale = filter_ratio*filter_width_scale(self.s, filter_width)
        self.delta = filter_ratio*_cartesian_filter_width(self.s, filter_width)

    def viscous_flux(self, U, t=0.):
        if self.sgs is None:
            return super().viscous_flux(U, t)
        gradient = self.viscous_operator.gradient(self.primitive(U), t)
        return self.eq.viscous_flux(U, gradient)+self.sgs.flux(self.eq, U, gradient, self.delta)

    def timestep(self, U, cfl=.1, *, t=0.):
        if self.sgs is None:
            return super().timestep(U, cfl)
        gradient = self.viscous_operator.gradient(self.primitive(U), t)
        mu_t = self.sgs.viscosity(U[..., 0], gradient, self.delta)
        # Conservative global coefficient bound, including SGS thermal diffusion.
        ratio = max(float((1+mu_t/self.mu).max()),
                    float((1+mu_t/self.mu*self.eq.prandtl/self.sgs.turbulent_prandtl).max()))
        return self.eq.timestep(self.s, U, cfl, viscous_factor=ratio)
