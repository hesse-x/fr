"""Channel geometry, states and independent small-grid FR/LDG reference physics.

Production advancement is implemented by GroupedFused and NativeLocalRKStepper.
"""
import math
import torch
from fr import CartesianFR, CartesianLDG, rusanov_flux
from tensor_ops import ExplicitEuler


class Channel:
    def __init__(self, cells=(64, 24, 32), p=3, re_tau=180., mach=.1,
                 device='cpu', forcing=1., stretch=1.2,
                 lengths=(2*math.pi, math.pi), bulk_velocity=15.):
        if not (math.isfinite(re_tau) and re_tau > 0 and
                math.isfinite(mach) and 0 < mach < 1 and
                math.isfinite(forcing) and forcing >= 0):
            raise ValueError('require Re_tau>0, 0<Mach<1, forcing>=0')
        if not math.isfinite(stretch) or not 0 <= stretch <= 3:
            raise ValueError('require wall-normal stretching in [0,3]')
        self.eq = ExplicitEuler(3)
        # h=rho_ref=u_tau_ref=1; Mach is based on reference velocity 15.
        self.mu = 1/re_tau
        self.temperature = (15/mach)**2/self.eq.gamma  # R=1
        self.forcing = forcing  # constant force per unit volume
        if len(lengths) != 2 or any(not math.isfinite(v) or v <= 0 for v in lengths):
            raise ValueError('two positive periodic domain lengths required')
        if not math.isfinite(bulk_velocity) or bulk_velocity <= 0:
            raise ValueError('positive bulk velocity required')
        self.bulk_velocity = bulk_velocity
        self.lengths = tuple(lengths)
        self.volume = 2*math.prod(lengths)
        self.kappa = self.mu*self.eq.gamma/(self.eq.gamma-1)/.71
        eta = torch.linspace(-1., 1., cells[1]+1, dtype=torch.float64, device=device)
        y_edges = torch.tanh(stretch*eta)/math.tanh(stretch) if stretch else eta
        self.s = CartesianFR(cells, ((0., lengths[0]), (-1., 1.), (0., lengths[1])),
                             p, flux=self.eq.flux,
                             numerical_flux=rusanov_flux(self.eq.speed),
                             speed=self.eq.speed, periodic=(True, False, True),
                             boundary=self.wall, device=device, edges=(None, y_edges, None))
        self.ldg = CartesianLDG(self.s, boundary=self.wall_primitive)
        self.reference_ldg = self.ldg
        # Independent Cartesian FR convection for small-grid correctness checks.
        # The native planner only consumes these constant reference fluxes.
        def convection(U, t=0., fields=None):
            return self.s.rhs(U, t)
        reference = self.s.z.new_tensor([1., 15., 0., 0.,
            self.temperature/(self.eq.gamma-1)+112.5])
        convection.flux_mode = 'local'
        convection.reference_flux = tuple(self.eq.flux(reference, a) for a in range(3))
        self.convection = convection

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
        rho, velocity, pressure = self.eq.primitive(U)
        return torch.cat((velocity, (pressure/rho)[..., None]), -1)

    def state(self, velocity, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        rho = torch.ones_like(velocity[..., :1])
        energy = rho*(temperature/(self.eq.gamma-1)+.5*velocity.square().sum(-1, keepdim=True))
        return torch.cat((rho, rho*velocity, energy), -1)


    def initial(self, amplitude=.5):
        x, y, z = self.s.coords
        # Smooth periodic, divergence-free velocity perturbation, zero at walls.
        envelope = (1-y*y)**2
        base = 1-y**8
        base = self.bulk_velocity*base/(self.s.integral(base[..., None])[0]/self.volume)
        kx, kz = 2*math.pi/self.lengths[0], 2*math.pi/self.lengths[1]
        u = base+amplitude*envelope*torch.cos(kz*z)
        v = amplitude*envelope*torch.sin(kx*x)*torch.cos(kz*z)
        w = (4/kz)*amplitude*y*(1-y*y)*torch.sin(kx*x)*torch.sin(kz*z)
        return self.state(torch.stack((u, v, w), -1))


    def viscous_flux(self, U, t=0.):
        primitive = self.primitive(U)
        grad = self.reference_ldg.gradient(primitive, t)
        du = grad[..., :3, :]
        stress = self.mu*(du+du.transpose(-1, -2)
                          -2/3*du.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
                          *torch.eye(3, dtype=U.dtype, device=U.device))
        energy = (primitive[..., :3, None]*stress).sum(-2)+self.kappa*grad[..., 3, :]
        return torch.cat((torch.zeros_like(energy[..., None, :]), stress,
                          energy[..., None, :]), -2)


    def rhs(self, U, t=0.):
        flux = self.viscous_flux(U, t)
        result = self.convection(U, t)+self.reference_ldg.divergence(flux, t)
        # Stationary walls do no mechanical work. LDG's default dual trace
        # uses interior flux; remove its velocity dot stress at the two walls.
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
        correction = (2/s.cell_width(1))*correction.permute([perm.index(i) for i in range(U.ndim)])
        result[..., -1:] += correction
        result[..., 1] += self.forcing
        result[..., -1] += self.forcing*U[..., 1]/U[..., 0]
        return result


    def timestep(self, U, cfl=.1):
        # Conservative explicit diffusion bound; acoustic bound from FR.
        density, velocity, pressure = self.eq.primitive(U)
        rho = float(density.min())
        diffusivity = max(4/3*self.mu, self.kappa*(self.eq.gamma-1))/rho
        viscous = cfl/((self.s.p+1)**4*sum(float(h.min())**-2 for h in self.s.widths)*diffusivity)
        sound = (self.eq.gamma*pressure/density).sqrt()
        rate = sum((velocity[..., a].abs()+sound)/self.s.cell_width(a, variables=False)
                   for a in range(3)).max()
        return min(cfl/((2*self.s.p+1)*float(rate)), viscous)


    def plane_mean(self, value):
        # (*cells, *nodes, variables) -> (y cell, y node, variables).
        weights = self.s.w
        return torch.einsum('abcdefg,d,f->beg', value, weights, weights)/(4*self.s.cells[0]*self.s.cells[2])


    def moments(self, U):
        velocity = self.primitive(U)[..., :3]
        products = torch.stack([velocity[..., i]*velocity[..., j]
                                for i, j in ((0, 0), (1, 1), (2, 2), (0, 1))], -1)
        return self.plane_mean(torch.cat((velocity, products), -1))


    def field_moments(self, U):
        rho, velocity, pressure = self.eq.primitive(U)
        products = torch.stack([velocity[..., i]*velocity[..., j]
                                for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], -1)
        return torch.cat((rho[..., None], velocity, pressure[..., None],
                          (pressure/rho)[..., None], products), -1)


    def perturb(self, U, amplitude, event):
        x, y, z = self.s.coords
        phase = event*2.399963229728653
        envelope = (1-y*y)**2
        kx, kz = 2*math.pi/self.lengths[0], 2*math.pi/self.lengths[1]
        wave = torch.sin(kx*x+phase)
        span = kz*z+phase
        delta = amplitude*torch.stack((envelope*torch.cos(span),
                                      envelope*wave*torch.cos(span),
                                      (4/kz)*y*(1-y*y)*wave*torch.sin(span)), -1)
        rho, velocity, _ = self.eq.primitive(U)
        result = U.clone()
        result[..., 1:4] += rho[..., None]*delta
        # Preserve density and internal energy/pressure while adding kinetic energy.
        result[..., -1] += rho*((velocity*delta).sum(-1)+.5*delta.square().sum(-1))
        return result
