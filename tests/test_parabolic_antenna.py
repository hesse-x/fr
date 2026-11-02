import unittest

import torch

from case.parabolic_antenna.solve import ParabolicAntennaCase


class ParabolicAntennaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_geometry_and_closed_surface(self):
        case = ParabolicAntennaCase(refinement=1)
        self.assertTrue(case.inside_reflector((torch.tensor(0.), torch.tensor(-.25), torch.tensor(0.))))
        self.assertFalse(case.inside_reflector((torch.tensor(0.), torch.tensor(1.), torch.tensor(0.))))
        closure = (case.surface_weights[:, None]*case.surface_normals).sum(0)
        self.assertLess(closure.norm().item(), 1e-12)
        self.assertTrue(bool(case.solid.any()))
        source = case.pml.physical(case.rhs(case.initial, 2.))
        self.assertAlmostEqual(case.integral(source)[2].item(), -1., places=11)
        self.assertFalse(bool((source.abs().sum(-1)[case.solid_nodes.expand(case.shape)] > 0).any()))

    def test_internal_pec_flux_and_frozen_solid(self):
        case = ParabolicAntennaCase(refinement=1)
        rhs = case.rhs(case.initial, 2.)
        self.assertEqual(rhs[case.solid_nodes.expand(case.shape)].abs().max().item(), 0.)
        self.assertTrue(bool(torch.isfinite(rhs).all()))

    def test_invalid_geometry(self):
        for kwargs in ({'diameter': 0}, {'focal_length': float('nan')},
                       {'refinement': 0}, {'thickness': .01}, {'radius': 1.}):
            with self.assertRaises(ValueError):
                ParabolicAntennaCase(**kwargs)


if __name__ == '__main__':
    unittest.main()
