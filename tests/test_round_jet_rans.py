import unittest

import numpy as np

from case.round_jet_rans import solve_round_jet


class RoundJetRANSTests(unittest.TestCase):
    def test_default_jet_spreads_and_conserves_axial_momentum(self):
        solution = solve_round_jet()
        self.assertTrue(np.all(np.isfinite(solution.nu_t)))
        self.assertTrue(np.all(solution.k > 0))
        self.assertTrue(np.all(solution.epsilon > 0))
        self.assertTrue(np.all(solution.radial_flux[:, 0] == 0))
        self.assertLess(solution.centerline_velocity[-1], solution.centerline_velocity[0]/2)
        self.assertGreater(solution.half_velocity_radius[-1], solution.half_velocity_radius[0]*2)

        area = 0.5*np.diff(solution.radial_faces**2)
        velocity = solution.axial_velocity
        mass_change = np.diff(velocity @ area)/np.diff(solution.x)
        np.testing.assert_allclose(mass_change, -solution.radial_flux[1:, -1], atol=1e-12)
        momentum = (velocity*(velocity-solution.coflow)) @ area
        np.testing.assert_allclose(momentum, momentum[0], rtol=1e-8)

    def test_domain_must_enclose_nozzle(self):
        with self.assertRaisesRegex(ValueError, 'radius must exceed'):
            solve_round_jet(radius=0.5)
