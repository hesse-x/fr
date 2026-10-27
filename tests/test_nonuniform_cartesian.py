import unittest
import torch

from fr import CartesianFR, CartesianLDG


class NonuniformCartesianTests(unittest.TestCase):
    def grid(self, periodic=False, **kwargs):
        return CartesianFR((3, 2), ((0., 1.), (0., 1.)), p=3,
                           flux=lambda U, axis: U,
                           flux_pair=lambda left, right: .5*(left+right),
                           periodic=periodic,
                           boundary=lambda coords, U, axis, side, t: sum(coords)[..., None],
                           edges=([0., .1, .4, 1.], [0., .7, 1.]), **kwargs)

    def test_geometry_integrals_and_linear_derivative(self):
        s = self.grid()
        x, y = s.coords
        U = (x+y)[..., None]
        self.assertAlmostEqual(float(s.integral(torch.ones_like(U))[0]), 1.)
        self.assertAlmostEqual(float(s.integral(U)[0]), 1.)
        self.assertLess(float((s.rhs(U)+2).abs().max()), 1e-11)
        self.assertTrue(torch.allclose(s.trace_coords[0][0][0][:, 0, 0], s.edges[0][:-1]))

    def test_periodic_mass_conservation(self):
        s = self.grid(periodic=True)
        U = torch.randn((*s.shape, 2), dtype=torch.float64)
        self.assertLess(float(s.integral(s.rhs(U)).abs().max()), 1e-11)

    def test_ldg_adjoint_and_diffusion(self):
        s = self.grid()
        ldg = CartesianLDG(s, boundary=lambda coords, U, axis, side, t: U*0)
        U = torch.randn((*s.shape, 1), dtype=torch.float64)
        grad = ldg.gradient(U)
        lhs = s.integral(U*ldg.divergence(grad))
        rhs = -s.integral(grad.square().sum(-1))
        self.assertTrue(torch.allclose(lhs, rhs, atol=1e-10, rtol=1e-12))

    def test_local_cfl(self):
        s = self.grid(speed=lambda U, axis: torch.ones_like(U[..., 0]))
        U = torch.ones((*s.shape, 1), dtype=torch.float64)
        self.assertAlmostEqual(s.timestep(U, .1), .1/(7*(10+1/.3)))

    def test_bad_edges(self):
        with self.assertRaises(ValueError):
            CartesianFR((2,), ((0., 1.),), flux=lambda U, a: U,
                        flux_pair=lambda l, r: l, periodic=True, edges=([0., .7, .6],))
