import unittest
import torch
from fr import Face, CartesianFR, integrate
from fr.acoustics import Acoustics
from fr.fluxes import roe_flux, LaxWendroffFlux
from fr.boundaries import acoustic_wall, per_face, extrapolate
from fr.pml import SplitPML
from case.acoustic_pml import AcousticPMLCase


class AcousticPMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_roe_absolute_jacobian(self):
        eq = Acoustics()
        jump = torch.tensor([2.,3.,4.],dtype=torch.float64)
        for axis in range(2):
            absolute = eq.absolute_jacobian_action(jump,None,None,axis)
            self.assertTrue(torch.allclose(eq.flux(eq.flux(jump,axis),axis),eq.c*absolute))

    def test_split_rhs_matches_unsplit_without_damping(self):
        eq = Acoustics()
        layer = SplitPML(eq,((-1.,1.),)*2,1.)
        face = roe_flux(eq.absolute_jacobian_action)
        plain = CartesianFR((2,2),((-.5,.5),)*2,flux=eq.flux,numerical_flux=face,periodic=True)
        split = CartesianFR((2,2),((-.5,.5),)*2,flux=layer.flux,numerical_flux=layer.numerical_flux(face),source=layer.source,periodic=True)
        torch.manual_seed(2)
        U = torch.randn(*plain.shape,3,dtype=torch.float64)
        self.assertTrue(torch.allclose(layer.physical(split.rhs(layer.pack(U))),plain.rhs(U),atol=1e-12))

    def test_wall_and_dispatch(self):
        U = torch.tensor([1.,2.,3.])
        bc = per_face({(0,-1): acoustic_wall})
        self.assertTrue(torch.equal(bc((),U,0,-1,0),torch.tensor([1.,-2.,3.])))
        self.assertTrue(torch.equal(bc((),U,1,1,0),U))

    def test_lw_truncated_step_context(self):
        flux = LaxWendroffFlux(lambda U,a: 2*U)
        m,p = Face(torch.tensor([1.]),torch.tensor([2.])),Face(torch.tensor([3.]),torch.tensor([6.]))
        with self.assertRaises(RuntimeError): flux(m,p,0)
        recorded=[]
        integrate(lambda U,t: U*0,torch.ones(1),.15,dt=.1,
                  prepare_step=lambda dt,t:(recorded.append(dt),flux.prepare_step(dt,(1.,))))
        self.assertAlmostEqual(recorded[-1],.05)
        self.assertAlmostEqual(flux(m,p,0).item(),3.8,places=6)

    def test_layer_reduces_returned_energy(self):
        absorb = AcousticPMLCase(8)
        wall = AcousticPMLCase(8,absorb=False)
        _, a = absorb.run(3.)
        _, b = wall.run(3.)
        self.assertLess(a['core_energy'], .02*b['core_energy'])

    def test_profile_and_damping(self):
        layer = SplitPML(Acoustics(), ((-1.,1.),)*2,1.)
        x = torch.tensor([0.,1.,2.],dtype=torch.float64)
        y = torch.zeros_like(x)
        rates = layer.rates((x,y))
        self.assertEqual(rates[0].sum().item(),0.)
        self.assertEqual(rates[1].sum().item(),0.)
        self.assertAlmostEqual(rates[2,0].item(),layer.sigma_max)
        self.assertEqual(rates[:,1].sum().item(),0.)
        Q = torch.ones(3,6,dtype=torch.float64)
        source = layer.source((x,y),Q,0.)
        self.assertTrue((source*Q <= 0).all())

    def test_pulse_generates_both_velocity_components(self):
        case = AcousticPMLCase(8)
        U,stats = case.run(.05)
        self.assertGreater(U[...,1].abs().max().item(),.01)
        self.assertGreater(U[...,2].abs().max().item(),.01)
        self.assertTrue(torch.isfinite(U).all())


if __name__ == '__main__': unittest.main()
