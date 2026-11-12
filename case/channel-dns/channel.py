"""Channel geometry, states and independent small-grid FR/LDG reference physics.

Production advancement is implemented by GroupedFused and NativeLocalRKStepper.
"""
import math
import torch
from fr import CartesianNavierStokes, CartesianMesh


class Channel(CartesianNavierStokes):
    def __init__(self, cells=(64, 24, 32), p=3, re_tau=180., mach=.1,
                 device='cpu', forcing=1., stretch=1.2,
                 lengths=(2*math.pi, math.pi), bulk_velocity=15., viscous_scheme='ldg', penalty=4., mesh=None):
        if not (math.isfinite(re_tau) and re_tau > 0 and
                math.isfinite(mach) and 0 < mach < 1 and
                math.isfinite(forcing) and forcing >= 0):
            raise ValueError('require Re_tau>0, 0<Mach<1, forcing>=0')
        if not math.isfinite(stretch) or not 0 <= stretch <= 3:
            raise ValueError('require wall-normal stretching in [0,3]')
        # h=rho_ref=u_tau_ref=1; Mach is based on reference velocity 15.
        temperature = (15/mach)**2/1.4  # R=1
        if len(lengths) != 2 or any(not math.isfinite(v) or v <= 0 for v in lengths):
            raise ValueError('two positive periodic domain lengths required')
        if not math.isfinite(bulk_velocity) or bulk_velocity <= 0:
            raise ValueError('positive bulk velocity required')
        self.bulk_velocity = bulk_velocity
        self.lengths = tuple(lengths)
        eta = torch.linspace(-1., 1., cells[1]+1, dtype=torch.float64, device=device)
        y_edges = torch.tanh(stretch*eta)/math.tanh(stretch) if stretch else eta
        edges = (None, y_edges, None) if mesh is None else CartesianMesh.read(mesh).require(
            cells, ((0., lengths[0]), (-1., 1.), (0., lengths[1])))
        super().__init__(cells, ((0., lengths[0]), (-1., 1.), (0., lengths[1])), p,
                         mu=1/re_tau, wall_temperature=temperature, forcing=forcing,
                         boundary='isothermal-channel', device=device, edges=edges,
                         reference_state=[1., 15., 0., 0., temperature/(1.4-1)+112.5],
                         viscous_scheme=viscous_scheme, penalty=penalty)

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
