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

    def absolute_jacobian_action(self, jump, left, right, axis):
        """Roe |A| jump, with Harten quadratic fix on acoustic waves only.

        Fix width is 0.05 times Roe sound speed; contact/shear waves retain |u_n|.
        Accepts arbitrary conservative jumps, not just right-left.
        """
        rl, ul, pl = self.primitive(left)
        rr, ur, pr = self.primitive(right)
        sl, sr = rl.sqrt(), rr.sqrt()
        velocity = (sl[..., None]*ul+sr[..., None]*ur)/(sl+sr)[..., None]
        hl, hr = (left[..., -1]+pl)/rl, (right[..., -1]+pr)/rr
        enthalpy = (sl*hl+sr*hr)/(sl+sr)
        q2 = velocity.square().sum(-1)
        sound2 = (self.gamma-1)*(enthalpy-.5*q2)
        sound = sound2.sqrt()
        normal = velocity[..., axis]
        drho, dm, de = jump[..., 0], jump[..., 1:-1], jump[..., -1]
        pressure_jump = (self.gamma-1)*(de-(velocity*dm).sum(-1)+.5*q2*drho)
        normal_jump = dm[..., axis]-normal*drho
        minus = (pressure_jump-sound*normal_jump)/(2*sound2)
        plus = (pressure_jump+sound*normal_jump)/(2*sound2)
        contact = drho-pressure_jump/sound2
        def acoustic(value):
            value = value.abs()
            width = .05*sound
            return torch.where(value < width, .5*(value.square()/width+width), value)
        lm, lp, l0 = acoustic(normal-sound), acoustic(normal+sound), normal.abs()
        am, ap, ac = lm*minus, lp*plus, l0*contact
        density = am+ac+ap
        direction = torch.zeros_like(velocity)
        direction[..., axis] = 1.
        shear = dm-velocity*drho[..., None]-direction*normal_jump[..., None]
        momentum = density[..., None]*velocity+direction*(sound*(ap-am))[..., None]+l0[..., None]*shear
        energy = am*(enthalpy-sound*normal)+ac*.5*q2+ap*(enthalpy+sound*normal)+l0*(velocity*shear).sum(-1)
        return torch.cat((density[..., None], momentum, energy[..., None]), -1)

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
