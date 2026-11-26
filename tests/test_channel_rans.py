import importlib
import unittest

import numpy as np

from fr import CartesianMesh


rans = importlib.import_module('case.channel-rans.solve')


class ChannelRANSTests(unittest.TestCase):
    def test_dns_grid_steady_turbulent_branch(self):
        mesh = CartesianMesh.read(rans.DNS_MESH)
        solution = rans.solve_channel(mesh)
        self.assertEqual(mesh.cells, (64, 24, 32))
        self.assertEqual(len(solution.y_over_h), 48)
        self.assertAlmostEqual(solution.y_over_h[0]*rans.DNS_RE_TAU, .49186048)
        self.assertLess(solution.residual, 1e-8)
        self.assertTrue(np.all(solution.k_plus > 0))
        self.assertTrue(np.all(solution.omega_plus > 0))
        self.assertGreater(solution.nu_t_over_nu.max(), 1)
        self.assertTrue(np.all(np.diff(solution.u_plus) > 0))
        self.assertGreater(solution.center_u_plus, solution.bulk_u_plus)
        np.testing.assert_allclose(
            solution.viscous_shear_plus+solution.reynolds_shear_plus, 1-solution.y_over_h,
            atol=2e-14,
        )

    def test_calibration_can_continue_from_standard_solution(self):
        mesh = CartesianMesh.read(rans.DNS_MESH)
        standard = rans.solve_channel(mesh)
        damping = rans.WilcoxDamping(r_k=5.5)
        fitted = rans.solve_channel(mesh, damping=damping, initial_solution=standard)
        direct = rans.solve_channel(mesh, damping=damping)
        np.testing.assert_array_equal(fitted.y_over_h, standard.y_over_h)
        np.testing.assert_allclose(fitted.u_plus, direct.u_plus)
        self.assertGreater(abs(fitted.bulk_u_plus-standard.bulk_u_plus), 0.01)

    def test_stress_limiter_preserves_the_momentum_relation(self):
        k = np.array([.1, .1])
        omega = np.array([.01, 200.])
        total_shear = np.array([.95, .95])
        nu = 1/180
        gradient, nu_t, diffusion, *_ = rans.wilcox_terms(k, omega, nu, total_shear)
        self.assertLess(nu_t[0], diffusion[0])
        self.assertAlmostEqual(nu_t[1], diffusion[1])
        np.testing.assert_allclose((nu+nu_t)*gradient, total_shear, atol=2e-14)
