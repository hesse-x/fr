"""Airfoil boundary data and loads over the framework NS viscous operators."""
import importlib.util
from pathlib import Path
import math
import torch
from fr import NavierStokes, make_viscous_operator

spec = importlib.util.spec_from_file_location('naca_euler_physics', Path(__file__).parents[1]/'naca0012-euler/airfoil.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Airfoil(module.Airfoil):
    def __init__(self, mesh, p=2, mach=.1, alpha=0., quadrature_order=6,
                 reynolds=1000., prandtl=.72, penalty=4., viscous_step_factor=1.,
                 viscous_scheme='br1-penalty'):
        if not all(math.isfinite(v) and v > 0 for v in (reynolds, prandtl, mach, viscous_step_factor)):
            raise ValueError('positive finite Re, Pr, Mach and viscous step factor required')
        super().__init__(mesh, p, mach, alpha, quadrature_order)
        self.reynolds, self.prandtl = reynolds, prandtl
        # rho_inf=c_inf=chord=1; Re=rho_inf*u_inf*chord/mu.
        self.mu = mach/reynolds
        self.eq = NavierStokes(2, mu=self.mu, prandtl=prandtl)
        self.viscous_step_factor = viscous_step_factor
        self.viscous_operator = make_viscous_operator(self.s, self.eq,
            scheme=viscous_scheme, penalty=penalty, boundary=self.viscous_boundary,
            normal_flux=self.viscous_boundary_flux)
        self.viscous_config = self.viscous_operator.config
        self.viscous_scheme, self.penalty = self.viscous_config.scheme, self.viscous_config.penalty
        self.h = self.viscous_operator.blocks[0]['h']
        self.face_h = self.viscous_operator.face_h
        # Native consumes the same framework geometry as the Torch operator.
        geometry = self.viscous_operator.blocks[0]
        self.Gx, self.Gy, self.lift_matrix = geometry['Gx'], geometry['Gy'], geometry['lift']
        b = self.s.blocks[0]
        self.lift = self.lift_matrix.reshape(b['ne'], b['ns'], b['ids'].shape[1], -1).permute(0, 2, 1, 3)

    def primitive_vector(self, U):
        return self.eq.primitive_variables(U)

    def viscous_boundary(self, coords, inside, normal, tag, t):
        if tag == 2:
            value = inside.clone()
            value[..., :2] = 0.  # no slip; interior theta for adiabatic wall
            return value
        return self.primitive_vector(self.far).expand_as(inside)

    @staticmethod
    def viscous_boundary_flux(coords, inside, normal, tag, t, flux):
        if tag == 2:
            flux = flux.clone()
            flux[..., -1] = 0.  # stationary adiabatic wall
        return flux

    def gradient(self, W):
        return self.viscous_operator.gradient(W)

    def viscous_flux(self, W, grad):
        return self.viscous_operator.flux(W, grad)

    def viscous(self, U):
        return self.viscous_operator.evaluate(U)

    def loads(self, U):
        cl, cd, pressure = super().loads(U)
        _, flux = self.viscous(U)
        ids = self.wall_ids; s = self.s
        force = -(flux[ids,:,1:3]*s.face_jacobian[ids,:,None]*s.fw[None,:,None]).sum((0,1))/(.5*self.mach**2)
        return cl+float(force@torch.stack((-self.direction[1],self.direction[0]))), cd+float(force@self.direction), pressure

    def wall_data(self, U):
        """Numerical traction consistent with momentum balance; tangent points toward +x."""
        s = self.s; ids = self.wall_ids
        clp, cdp, pressure = super().loads(U)
        _, flux = self.viscous(U)
        traction = -flux[ids,:,1:3]
        n = s.normals[ids]
        orient = torch.where(n[...,1]>=0,1.,-1.)
        tangent = torch.stack((n[...,1],-n[...,0]),-1)*orient[...,None]
        q = .5*self.mach**2
        cf = (traction*tangent).sum(-1)/q
        force = (traction*s.face_jacobian[ids,:,None]*s.fw[None,:,None]).sum((0,1))/q
        cdv = float(force@self.direction)
        clv = float(force@torch.stack((-self.direction[1],self.direction[0])))
        return dict(CL=clp+clv,CD_total=cdp+cdv,CL_pressure=clp,CL_viscous=clv,
                    CD_pressure=cdp,CD_viscous=cdv), torch.cat((s.face_coordinates[ids],
                    ((pressure-1/1.4)/q)[...,None],cf[...,None],
                    s.normals[ids],traction/q),-1).reshape(-1,8)
