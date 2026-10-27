import unittest
import torch
from channel import Channel


class ChannelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = Channel((2, 3, 2), p=3)


    def test_boundary_and_mass(self):
        m = self.model
        self.assertEqual(m.s.periodic, (True, False, True))
        U = m.initial()
        self.assertTrue(m.eq.valid(U))
        self.assertLess(abs(float(m.s.integral(m.rhs(U))[0])), 1e-10)
        self.assertAlmostEqual(float(m.s.integral(U)[1]/m.s.integral(U)[0]), 15.)
        for side in (-1, 1):
            value = m.wall_primitive(None, m.primitive(U), 1, side, 0.)
            self.assertEqual(float(value[..., :3].abs().max()), 0.)
            self.assertTrue(torch.all(value[..., 3] == m.temperature))


    def test_laminar_momentum_balance(self):
        # Energy contains u^2, a quartic polynomial; represent it exactly.
        m = Channel((2, 3, 2), p=5)
        x, y, z = m.s.coords
        u = (1-y*y)/(2*m.mu)
        U = m.state(torch.stack((u, torch.zeros_like(x), torch.zeros_like(z)), -1))
        # Exact Poiseuille momentum balance: mu*u_yy + force = 0.
        self.assertLess(float(m.rhs(U)[..., 1:4].abs().max()), 1e-8)


    def test_rest_equilibrium(self):
        m = Channel((2, 3, 2), p=2, forcing=0.)
        U = m.state(torch.zeros((*m.s.shape, 3), dtype=torch.float64))
        self.assertLess(float(m.rhs(U).abs().max()), 1e-7)


    def test_plane_mean(self):
        m = self.model
        mean = m.plane_mean(torch.ones((*m.s.shape, 2), dtype=torch.float64))
        self.assertTrue(torch.allclose(mean, torch.ones_like(mean)))


    def test_repeated_perturbation_preserves_thermodynamics(self):
        m = self.model
        U = m.initial()
        perturbed = m.perturb(U, .5, 2)
        rho, _, pressure = m.eq.primitive(U)
        new_rho, _, new_pressure = m.eq.primitive(perturbed)
        self.assertTrue(torch.equal(rho, new_rho))
        self.assertTrue(torch.allclose(pressure, new_pressure, atol=1e-10, rtol=1e-12))


    def test_optimized_cfl_matches_original_bound(self):
        m = self.model
        U = m.initial()
        diffusion = max(4/3*m.mu, m.kappa*(m.eq.gamma-1))/float(U[..., 0].min())
        viscous = .1/((m.s.p+1)**4*sum(float(w.min())**-2 for w in m.s.widths)*diffusion)
        self.assertAlmostEqual(m.timestep(U), min(m.s.timestep(U, .1), viscous))


    def test_profile_from_full_field_moments(self):
        m = self.model
        U = m.initial()
        V = m.perturb(U, .5, 3)
        integral = torch.zeros_like(m.field_moments(U))
        integral.add_(m.field_moments(U), alpha=.03)
        integral.add_(m.field_moments(V), alpha=.03)
        actual = m.plane_mean(integral[..., [1, 2, 3, 6, 7, 8, 9]])
        expected = .03*(m.moments(U)+m.moments(V))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))



if __name__ == '__main__':
    unittest.main()
