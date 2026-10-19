import math
import unittest
import torch
from fr import CartesianFR, integrate
from fr.acoustics import Acoustics, ConvectedAcoustics
from fr.fluxes import roe_flux
from case.acoustic_pml import AcousticPMLCase
from case.acoustic_mean_flow_pml import MeanFlowPMLCase


class MeanFlowPMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_characteristics_include_transverse_advection(self):
        eq = ConvectedAcoustics((.3,-.2),rho=2.,c=1.4)
        for axis in range(2):
            for sign in (-1,1):
                vector = torch.zeros(3,dtype=torch.float64)
                vector[0],vector[axis+1] = eq.rho*eq.c, sign
                speed = eq.mean[axis]+sign*eq.c
                self.assertTrue(torch.allclose(eq.flux(vector,axis),speed*vector))
                action = eq.absolute_jacobian_action(vector,None,None,axis)
                self.assertTrue(torch.allclose(action,abs(speed)*vector))
            tangent = torch.zeros(3,dtype=torch.float64)
            tangent[2-axis] = 1.
            self.assertTrue(torch.allclose(eq.absolute_jacobian_action(tangent,None,None,axis),abs(eq.mean[axis])*tangent))

    def test_fixed_grid_oblique_plane_wave(self):
        eq = ConvectedAcoustics((.3,.2))
        mesh = CartesianFR((6,6),((0.,1.),)*2,p=3,flux=eq.flux,
                           numerical_flux=roe_flux(eq.absolute_jacobian_action),periodic=True,speed=eq.speed)
        x,y = mesh.coords
        vector = torch.tensor([1.,1/math.sqrt(2),1/math.sqrt(2)],dtype=torch.float64)
        def exact(t):
            return torch.sin(2*math.pi*(x+y-(.5+math.sqrt(2))*t))[...,None]*vector
        U,_ = integrate(mesh.rhs,exact(0.),.05,dt=mesh.timestep(exact(0.)))
        self.assertLess(mesh.norm(U-exact(.05)).max().item(),.001)

    def test_zero_mean_recovers_existing_pml(self):
        original = AcousticPMLCase(6,mean=(0.,0.))
        fixed = MeanFlowPMLCase(6,mean=(0.,0.),thickness=1.,extent=2.)
        A,_ = original.run(.05)
        B,stats = fixed.run(.05)
        self.assertTrue(torch.allclose(A,B,atol=1e-13,rtol=1e-13))
        self.assertIn('fixed',stats['frame'])
        self.assertTrue(stats['finite'])

    def test_complete_linearized_euler_matches_isentropic_case(self):
        acoustic = MeanFlowPMLCase(6,thickness=1.,extent=2.)
        full = MeanFlowPMLCase(6,thickness=1.,extent=2.,model='linearized-euler')
        A,_ = acoustic.run(.05)
        B,_ = full.run(.05)
        self.assertEqual(B.shape[-1],4)
        torch.testing.assert_close(full.acoustic_state(B),A,atol=1e-13,rtol=1e-13)

    def test_nonzero_mean_changes_rhs_and_cfl(self):
        original = AcousticPMLCase(6)
        fixed = MeanFlowPMLCase(6,thickness=1.,extent=2.)
        self.assertFalse(torch.allclose(original.mesh.rhs(original.initial),fixed.mesh.rhs(fixed.initial)))
        self.assertLess(fixed.timestep(),original.timestep())
        self.assertTrue(torch.equal(original.layer.rates(original.mesh.coords),fixed.layer.rates(fixed.mesh.coords)))


if __name__ == '__main__': unittest.main()
