"""Native tensor overintegration: modes, fluxes, conservation and fused RK."""
import math
import unittest
import torch
from case.taylor_green_vortex_les import TaylorGreenVortexLES
from fr import CartesianNavierStokes


class NativeDealiasingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_modes_and_solvers(self):
        cases = [('roe', 'flux', 5), ('roe', 'surf-flux', 5),
                 ('roe', 'flux, surf-flux', 5), ('roe', 'flux, surf-flux', 6),
                 ('rusanov', 'flux, surf-flux', 6)]
        for flux, mode, nq in cases:
            with self.subTest(flux=flux, mode=mode, nq=nq):
                model = TaylorGreenVortexLES((5, 3, 2), interface_flux=flux,
                                             anti_alias=mode, quadrature_order=nq)
                state = model.initial()
                native = model.make_stepper('native-p3', block=(2, 2, 2))
                torch.testing.assert_close(native.plan.convection(state), model.convection(state), atol=3e-12, rtol=1e-10)
                torch.testing.assert_close(native.plan(state), model.rhs(state), atol=3e-12, rtol=1e-10)
                torch.testing.assert_close(model.s.integral(native.plan(state)), state.new_zeros(5), atol=3e-12, rtol=0)
                reference = model.make_stepper('torch')
                actual = expected = state
                dt = model.timestep(state, .1)
                for step in range(3):
                    actual, valid, _ = native.advance_finish(actual, step*dt, dt, .1)
                    expected = reference(expected, step*dt, dt)
                    self.assertTrue(valid)
                    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-13)
                self.assertEqual(native.plan.metadata['quadrature_order'], nq)
                self.assertEqual(native.plan.metadata['anti_alias'], sorted(model.s.anti_alias))

    def test_nonuniform_channel_and_boundary_sweep(self):
        model = CartesianNavierStokes((4, 2, 3), ((-math.pi, math.pi),(-1.,1.),(-math.pi,math.pi)),
                    edges=([-math.pi,-1.,0.,.6,math.pi],[-1.,-.2,1.],[-math.pi,-.6,0.,math.pi]),
                    boundary='isothermal-channel', wall_temperature=2.5, interface_flux='roe',
                    anti_alias='flux, surf-flux', quadrature_order=6)
        x,y,z = model.s.coords
        rho = 1+.01*torch.cos(x)*torch.cos(z)
        velocity = torch.stack((.1*(1-y*y),.02*torch.sin(x)*(1-y*y),torch.zeros_like(x)),-1)
        state = model.soa(model.eq.conservative(rho,velocity,2.5*rho))
        for mode in ('separate','fused'):
            native=model.make_stepper('native-p3',block=(2,1,2),boundary_mode=mode)
            torch.testing.assert_close(native.plan(state),model.rhs(state),atol=3e-12,rtol=1e-10)

    def test_constant_state(self):
        model = TaylorGreenVortexLES((3, 2, 2), interface_flux='roe', anti_alias='flux, surf-flux', quadrature_order=6)
        state = model.initial()
        state[...] = state.new_tensor([1.,.2,-.1,.3,20.])
        native = model.make_stepper('native-p3')
        torch.testing.assert_close(native.plan(state), torch.zeros_like(state), atol=2e-12, rtol=0)
